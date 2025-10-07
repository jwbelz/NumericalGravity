#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/matrix_free.templates.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/hdf5.h>

#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include <deal.II/matrix_free/operators.h>


namespace Scalar_Evolution
{
  using namespace dealii;

  /*
    Here are a bunch of parameters for controlling the simulation.
    Not good practice but I have all the solution vectors as global variables.
    This is was a lazy way so they didn't need to all be passed to all the operators.
    In all reality all the new and old and coarse and fine vectors should be inside the EvolutionProblem class.
    
    Quick run through of all the parameters listed below:
    Keep the dimensions at 1 as this was only build for 1D
    fe_degree is the degree of the finite element used. If you change this make sure to change the lsrk_scheme
      to a matching order of convergence. Found in time_a_alpha_integrators.h
    n_q_points nad alt_q_points are number of gaussian quadrature points. Shouldn't need to ever be changed.
    n_global_refinements is just the initial number of global refinements so that the mesh can start to "see"
      how large of errors there are.
    max_refinement_level is the maximum number of refinements that can be done on an individual cell
    outer and inner boundary sets the limits of the domain. Inner boundary is always 0.
    individual_cell_error_limit is the error limit before a cell is refined.
    IC_individual_cell_error_limit is the error limit for the initial conditions (usually true error is around
      4 orders of magnitude smaller)
    output_spacing is the number of time steps between each output of the solution
    temp_max_step is the maximum number of time steps that can be taken
    final_time is the maximum time of a central observer before the simulation is killed
    error_checking_spacing is the number of time steps between each error check (the code does full step and
      views the error. If the error is too large the code will undo the step and refine and try again)
    cfl_factor is the Courant number for the time step size
    seed amp,sigma, and rb are the amplitude, sigma, and radial offset of the initial conditions of
      the scalar field (it is a gaussian)
      seed amp has been changed to be taken in from the command line so this really just defines it.
    serial is needed just for critical point searches and is taken in from command line.
    uniform_refinement is a switch to turn off adaptive refinement and make it uniform (I don't think uniform currently works)
    convergence_test_switch is a switch to turn on convergence testing of evolution (it creates a vector
      for the error according to the equation for the evolution of a and outputs it in the vtu files)
    convergence_test_number is the number of time steps before it starts to calculate the convergence error
      (should be 5 or more)
    dof_r_val is a vector of the radial values of the degrees of freedom (used for the a and alpha solvers)
    dof_points_coarse and dof_points_fine are the points of the degrees of freedom (used for the initial conditions
      convergence testing)
    initial_convergence_switch is a switch to turn on convergence testing of the initial conditions
    coarse, fine, and true cycle are the cycles chosen for the initial conditions convergence testing

    All the analysis booleans are switches to turn on and off outputting of certain data for analysis.
    
    Then a whole bunch of global variables for the solution vectors explained above.

  */
  constexpr unsigned int dimension            = 1;
  constexpr unsigned int fe_degree            = 3;
  constexpr unsigned int n_q_points_1d        = fe_degree + 1;
  constexpr unsigned int alt_q_points         = fe_degree + 1;
  constexpr unsigned int n_global_refinements = 3;
  constexpr int max_refinement_level          = 24;
  constexpr double outer_boundary_value       = 40.;
  constexpr double inner_boundary             = 0.;
  constexpr double individual_cell_error_limit= 1e-8;
  constexpr double IC_individual_cell_error_limit = 1e-7;
  constexpr unsigned int output_spacing       = 100;
  constexpr unsigned int temp_max_step        = 2000000;
  constexpr double final_time                 = 7.0;
  constexpr unsigned int error_checking_spacing = 1;
  constexpr double cfl_factor                 = 0.3;
  /*constexpr*/ double seed_amp;
  double approx_alpha;
  constexpr double seed_sigma                 = 5.35;
  constexpr double seed_rb                    = 0.;
  int               serial;
  constexpr bool uniform_refinement           = false;

  constexpr bool convergence_test_switch      = true;
  constexpr unsigned int convergence_test_number = 5;

  constexpr bool phi_origin_analysis          = true;
  constexpr bool dof_time_analysis            = false;
  constexpr bool mesh_error_analysis          = false;

  std::vector<dealii::Point<dimension>>dof_r_val;
  std::vector<dealii::Point<dimension>>dof_points_coarse;
  std::vector<dealii::Point<dimension>>dof_points_fine;
  constexpr bool initial_convergence_switch   = false;
  constexpr unsigned int coarse_cycle         = 3;
  constexpr unsigned int fine_cycle           = coarse_cycle +1;
  constexpr unsigned int true_cycle           = 8;

  using Number = double;


  LinearAlgebra::distributed::Vector<double> psi_solution;
  LinearAlgebra::distributed::Vector<double> psi_coarse_solution;
  LinearAlgebra::distributed::Vector<double> psi_fine_solution;
  LinearAlgebra::distributed::Vector<double> new_psi_solution;
  LinearAlgebra::distributed::Vector<double> pi_solution;
  LinearAlgebra::distributed::Vector<double> new_pi_solution;
  LinearAlgebra::distributed::Vector<double> phi_solution;
  LinearAlgebra::distributed::Vector<double> phi_coarse_solution;
  LinearAlgebra::distributed::Vector<double> phi_fine_solution;
  LinearAlgebra::distributed::Vector<double> new_phi_solution;

  LinearAlgebra::distributed::Vector<double> a_solution;
  LinearAlgebra::distributed::Vector<double> new_a_solution;
  LinearAlgebra::distributed::Vector<double> a_coarse_solution;
  LinearAlgebra::distributed::Vector<double> a_fine_solution;
  LinearAlgebra::distributed::Vector<double> other_a_solution;
  LinearAlgebra::distributed::Vector<double> alpha_solution;
  LinearAlgebra::distributed::Vector<double> alpha_coarse_solution;
  LinearAlgebra::distributed::Vector<double> alpha_fine_solution;
  // List all the other solution vectors_
  LinearAlgebra::distributed::Vector<double> modal_solution;
  LinearAlgebra::distributed::Vector<double> old_pi_solution;
  LinearAlgebra::distributed::Vector<double> old_psi_solution;
  LinearAlgebra::distributed::Vector<double> old_phi_solution;
  LinearAlgebra::distributed::Vector<double> old_a_solution;
  LinearAlgebra::distributed::Vector<double> old_alpha_solution;

  Vector<double>                             constraint_violation;

    LinearAlgebra::distributed::Vector<double> old_2_a_solution;
    LinearAlgebra::distributed::Vector<double> old_3_a_solution;
    LinearAlgebra::distributed::Vector<double> old_4_a_solution;
    LinearAlgebra::distributed::Vector<double> old_2_alpha_solution;
    LinearAlgebra::distributed::Vector<double> old_3_alpha_solution;
    LinearAlgebra::distributed::Vector<double> old_4_alpha_solution;
    LinearAlgebra::distributed::Vector<double> old_2_psi_solution;
    LinearAlgebra::distributed::Vector<double> old_2_pi_solution;
    LinearAlgebra::distributed::Vector<double>                             convergence_violation;
    LinearAlgebra::distributed::Vector<double>                             convergence_cell_violation;

    bool output_diagnostics = false;
    bool output_prints = false;

    // This is for speeding up the writing output of diagnostics for
    // Cole's critical point search
    struct Diagnostics
    {
      std::vector<double> times;
      std::vector<double> masses;
      std::vector<double> alphas;

      void push(const double &time, const double &mass, const double &alpha)
      {
        times.push_back(time);
        masses.push_back(mass);
        alphas.push_back(alpha);
      }
      void flush()
      {
        std::string filename1 = "Mass-" + std::to_string(fe_degree) + "-" + std::to_string(serial);
        std::string filename2 = "Al-" + std::to_string(fe_degree) + "-" + std::to_string(serial);
        std::ofstream file1(filename1);
        std::ofstream file2(filename2);
        for (unsigned int i = 0; i < times.size(); ++i)
        {
          std::string str1 = std::to_string(times[i]) + " " + std::to_string(masses[i]) + "\n";
          std::string str2 = std::to_string(times[i]) + " " + std::to_string(alphas[i]) + "\n";
          file1 << str1;
          file2 << str2;
        }
        file1.close();
        file2.close();
      }
    };
}

// All the headers made by Sean Johnson that include all the operators
// and all the helper functions used throughout.
// The order in which the include files are listed does matter.
#include "helper_functions.h"
#include "fake_a_operator.h"
#include "fake_alpha_operator.h"
#include "phi_operator.h"
#include "pi_operator.h"
#include "psi_operator.h"
#include "real_a_operator.h"
#include "OE_operator.h"
#include "time_a_alpha_integrators.h"

namespace Scalar_Evolution
{
  using namespace dealii;
  // This is the engine for moving all the evolution along
  // It builds the grid, solves the initial conditions,
  // then solve the evolution eqns
  // using a multistage approach where on each intermediary step or stage
  // the constraint equations are solved.
  template <int dim>
  class EvolutionProblem
  {
  public:
    EvolutionProblem();

    void run(struct Diagnostics &diag);

  private:

    void make_grid();

    void make_constraints();

    void make_dofs();

    void output_results_IC(const unsigned int result_number);

    void output_results_IC_special(const unsigned int result_number);

    void output_results_Evolve(const unsigned int result_number);

    void IC_gen();

    void Hamiltonian_Violation(double &error_estimate);

    void IC_refinement(double &error_estimate, unsigned int &cycle);

    void a_filler();

    void convergence_test(const double &dt);

    void initial_convergence_test();

    void stop_check(struct Diagnostics &diag, unsigned int &time_step_number);

    double MassFinder();


    ConditionalOStream pcout;

  /*#ifdef DEAL_II_WITH_P4EST
    parallel::distributed::Triangulation<dim> triangulation;
  #else*/
    Triangulation<dim> triangulation;
  //#endif

    FESystem<dim>   fe_DG;
    FE_DGP<dim> fe_modal;
    MappingQ<dim>   mapping;
    DoFHandler<dim> dof_handler_DG;
    DoFHandler<dim> dof_handler_modal;

    AffineConstraints<double>            constraints_DG;

    fakeAOperator<dim,fe_degree,n_q_points_1d> fake_a_operator;
    fakeAlphaOperator<dim,fe_degree,n_q_points_1d> fake_alpha_operator;
    Phi_Operator<dim,fe_degree,n_q_points_1d> phi_operator;
    Pi_Operator<dim, fe_degree, n_q_points_1d> pi_operator;
    Psi_Operator<dim, fe_degree, n_q_points_1d> psi_operator;
    OE_Operator<dim, fe_degree, n_q_points_1d> oe_operator;
    A_Operator<dim, fe_degree, n_q_points_1d> a_operator;

    double time, time_step;
    double outer_boundary = outer_boundary_value;
  };

  // The constructor for this class just initializes all the variables
  // sets the initial time and time step to zero
  // I was having some problems with how P4EST was installed on
  // my computer so I commented the code out but it should work
  template <int dim>
  EvolutionProblem<dim>::EvolutionProblem()
    : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
/*  #ifdef DEAL_II_WITH_P4EST
    , triangulation(MPI_COMM_WORLD,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<
                      dim>::construct_multigrid_hierarchy)
  #else*/
    , triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
//  #endif
    , fe_DG(FE_DGQ<dim>(fe_degree))
    , fe_modal(fe_degree)
    , mapping(fe_degree)
    , dof_handler_DG(triangulation)
    , dof_handler_modal(triangulation)
    , fake_a_operator()
    , fake_alpha_operator()
    , phi_operator()
    , pi_operator()
    , psi_operator()
    , oe_operator()
    , a_operator()
    , time(0)
    , time_step(0)
  {}

  
  template <int dim>
  void EvolutionProblem<dim>::make_grid()
  {
    // This makes the mesh and sets boundary id's for outer edges
    // Origin should have a boundary id of 0
    // Outer edge should have a boundary id of 1

    // Make grid
    GridGenerator::hyper_cube(triangulation,
                                   inner_boundary,
                                   outer_boundary);

    // Initial refinements so adaptive mesh works
    triangulation.refine_global(n_global_refinements);
    // Setting boundary IDs
    for (const auto &cell : triangulation.cell_iterators())
    {
      for (const auto &face : cell->face_iterators())
        {
          const auto center = face->center();

          if (std::fabs(center(0) - (outer_boundary)) < 1e-12)
            face->set_boundary_id(1);
          else if  (std::fabs(center(0)) < 1e-12)
            face->set_boundary_id(0);
        }
      }
    }

    template <int dim>
    void EvolutionProblem<dim>::make_constraints()
    {
      /*
      This function just refills the affine constraints. Really not needed for
      1D. Just left in out of good practice. But no hanging nodes in 1D and no
      Dirichlet boundary conditions are enforced for the 1D spherically
      symmetric massless scalar field using these constraints.

      Also not really needed for DG in general. The constraints are just empty
      */
      const IndexSet locally_relevant_dofs_DG =
        DoFTools::extract_locally_relevant_dofs(dof_handler_DG);

      constraints_DG.clear();
      constraints_DG.reinit(locally_relevant_dofs_DG);
      constraints_DG.close();
    }

    template <int dim>
    void EvolutionProblem<dim>::make_dofs()
    {
    /*
    Function just reinitializes the matrix free operators with the new size of
    dofs or just to initialize in the first place.
    std::vector's are overkill for dof_handles and constriants_list but its
    from another code I made where they were more necessary.
    */
   // Setting up the list of dof_handlers and constraints that the operators will use
    const std::vector<const DoFHandler<dim> *> dof_handlers = {&dof_handler_DG};
    const std::vector<const AffineConstraints<double> *> constraints_list = {&constraints_DG};

   // Initializing the operators so they can make a matrix free object to 
   // interpret the solution vectors.
    phi_operator.reinit(mapping,dof_handlers,constraints_list);
    psi_operator.reinit(mapping,dof_handlers,constraints_list);
    pi_operator.reinit(mapping, dof_handlers, constraints_list);
    a_operator.reinit(mapping,dof_handlers,constraints_list);

    // This is just an output to the terminal so the user knows some basic
    // information about the mesh, elements, and degrees of freedom.
    if (output_prints)
    {
    std::locale s = pcout.get_stream().getloc();
    pcout.get_stream().imbue(std::locale(""));
    pcout << "Number of degrees of freedom: " << dof_handler_DG.n_dofs()
          << " ( = " << " [vars] x "
          << triangulation.n_global_active_cells() << " [cells] x "
          << Utilities::pow(fe_degree + 1, dim) << " [dofs/cell/var] )"
          << std::endl;
    pcout.get_stream().imbue(s);
    }
  }


  template <int dim>
  void EvolutionProblem<dim>::output_results_IC(const unsigned int result_number)
  {
    /*
    Function outputs the vtu files of the Initial Data after the solutions have
    been solved on new refined meshes.

    Takes in a result number which is just used in numbering output files.
    */

    {

      DataOut<dim>  data_out;
      // dof_handler attached and then solution vectors with labelling names
      data_out.attach_dof_handler(dof_handler_DG);
      data_out.add_data_vector(psi_solution,"psi",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(phi_solution, "phi",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(pi_solution, "pi",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(a_solution,"a",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(alpha_solution,"alpha");

      // Down here are vectors that are cell data instead of dof_data
      Vector<double> mpi_owner(triangulation.n_active_cells());
      mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      data_out.add_data_vector(mpi_owner, "owner",DataOut<dim>::type_cell_data);
      data_out.build_patches(mapping,
                             fe_DG.degree/*,
                             DataOut<dim>::curved_inner_cells*/);

      // Outputs the vtu file to folder outputs with the name Initial_###.vtu
      const std::string filename =
        "./outputs/Initial_" + Utilities::int_to_string(result_number, 3) + ".vtu";

      data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);

    }
  }

  template <int dim>
  void EvolutionProblem<dim>::output_results_IC_special(const unsigned int result_number)
  {

    /*
    This puts out vtu files for visualization with paraview for when doing
    convergence analysis of initial conditions. Not really needed anymore.

    Takes in a result number which is just used in numbering output files.
    */

    {

      DataOut<dim>  data_out;

      data_out.attach_dof_handler(dof_handler_DG);

      data_out.add_data_vector(a_solution,"a",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(a_coarse_solution,"a_coarse");
      data_out.add_data_vector(a_fine_solution,"a_fine");
      data_out.add_data_vector(alpha_solution,"alpha");
      data_out.add_data_vector(alpha_coarse_solution,"alpha_coarse");
      data_out.add_data_vector(alpha_fine_solution,"alpha_fine");
      data_out.add_data_vector(phi_solution,"phi");
      data_out.add_data_vector(phi_coarse_solution,"phi_coarse");
      data_out.add_data_vector(phi_fine_solution,"phi_fine");
      data_out.add_data_vector(psi_solution,"psi");
      data_out.add_data_vector(psi_coarse_solution,"psi_coarse");
      data_out.add_data_vector(psi_fine_solution,"psi_fine");

      Vector<double> mpi_owner(triangulation.n_active_cells());
      mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      data_out.add_data_vector(mpi_owner, "owner",DataOut<dim>::type_cell_data);
      data_out.build_patches(mapping,
                             fe_DG.degree/*,
                             DataOut<dim>::curved_inner_cells*/);

      const std::string filename =
        "./outputs/Initial_Special_" + Utilities::int_to_string(result_number, 3) + ".vtu";

      data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);

    }
  }

  template <int dim>
  void EvolutionProblem<dim>::output_results_Evolve(const unsigned int result_number)
  {
    /*
    Function outputs vtu file for visualization with paraview of the results after
    the initial data.

    Takes in a result number which is just used in numbering output files.
    */

    {

      DataOut<dim>  data_out;
/*
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out.set_flags(flags);
*/
      // Sets up dof_handler and all the degree of freedom type data
      data_out.attach_dof_handler(dof_handler_DG);
      data_out.add_data_vector(psi_solution,"psi",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(phi_solution, "phi",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(pi_solution, "pi",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(a_solution,"a",DataOut<dim>::type_dof_data);
      data_out.add_data_vector(alpha_solution,"alpha");
      //data_out.add_data_vector(other_a_solution,"other_a");
      if(convergence_test_switch && result_number*output_spacing > convergence_test_number){
        data_out.add_data_vector(convergence_violation, "convergence_violation");
      }

      // Now adds cell type data
      Vector<double> mpi_owner(triangulation.n_active_cells());
      mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      data_out.add_data_vector(mpi_owner, "owner");
      data_out.add_data_vector(constraint_violation,"Constraint_Violation");
      data_out.add_data_vector(convergence_cell_violation,"Convergence_Cell_Violation");
      data_out.build_patches(mapping,
                             fe_DG.degree/*,
                             DataOut<dim>::curved_inner_cells*/);

      // Outputs the vtu file to folder outputs with the name Evolutions_###.vtu
      const std::string filename =
        "./outputs/Evolutions_" + Utilities::int_to_string(result_number, 4) + ".vtu";

      data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
    }
  }

  template <int dim>
  void EvolutionProblem<dim>::IC_gen()
  {
    /*
    Function generates Initial Condition data. Fills phi and psi first  using
    functions defined in helper_functions.h and sets
    pi to zero. Then solves for a and alpha in the guzman way (a=1 at origin)
    integrate outwards then integrate alpha inwards (alpha = 1/a at far boundary)
    */

    make_constraints();
    // The renumber directions is used to order degrees of freedom
    // such that they start at the origin and increase moving outwards
    // in sequential order. Needed to simplify solving for a and alpha
    Tensor<1,dim> renumbering_direction;
    renumbering_direction[0] = 1.;

    dof_handler_DG.distribute_dofs(fe_DG);

    unsigned int cycle = 0;
    {
      // This is an old part of code that is still needed for it to work
      // the integrator is given lsrk_scheme but it really uses
      // RK4 within the fake_a and fake_alpha operators
      const LowStorageRungeKuttaIntegrator integrator(lsrk_scheme);

      // Operators are initialized and then used to initialize
      // the solution vectors
      fake_a_operator.reinit(mapping,dof_handler_DG,constraints_DG);
      fake_a_operator.initialize_vector(a_solution);
      fake_a_operator.initialize_vector(old_a_solution);
      fake_a_operator.initialize_vector(old_2_a_solution);
      fake_a_operator.initialize_vector(old_3_a_solution);
      fake_a_operator.initialize_vector(old_4_a_solution);

      fake_alpha_operator.reinit(mapping,dof_handler_DG,constraints_DG);
      fake_alpha_operator.initialize_vector(alpha_solution);
      fake_a_operator.initialize_vector(old_alpha_solution);
      fake_a_operator.initialize_vector(old_2_alpha_solution);
      fake_a_operator.initialize_vector(old_3_alpha_solution);
      fake_a_operator.initialize_vector(old_4_alpha_solution);

      fake_a_operator.fill_shape_vals(fe_DG);
      fake_alpha_operator.fill_shape_vals(fe_DG);

    // This vector holds the r values of the degrees of freedom.
    // It is first created and filled here
     dof_r_val.resize(dof_handler_DG.n_dofs());

     DoFTools::map_dofs_to_support_points(mapping,dof_handler_DG,dof_r_val);

      // the initial error is set large enough to start the loop
      double IC_error_estimate = 1.0;
      while(IC_error_estimate > IC_individual_cell_error_limit)
      {
        // After the first iteration/cycle this if statement
        // does the refining, renumbering of degrees of freedom,
        // and setting of boundary ids
        if (cycle > 0)
        {
          IC_refinement(IC_error_estimate,cycle);
          DoFRenumbering::downstream(dof_handler_DG, renumbering_direction);
          for (const auto &cell : triangulation.cell_iterators())
          {
            for (const auto &face : cell->face_iterators())
              {
                const auto center = face->center();

                if (std::fabs(center(0) - (outer_boundary)) < 1e-12)
                  face->set_boundary_id(1);
                else if  (std::fabs(center(0)) < 1e-12)
                  face->set_boundary_id(0);
              }
          }
        }
      // This reinitialized the operators and all solution vectors
      // in case the mesh has been refined.
      fake_a_operator.reinit(mapping,dof_handler_DG,constraints_DG);
      fake_a_operator.initialize_vector(a_solution);

      fake_alpha_operator.reinit(mapping,dof_handler_DG,constraints_DG);
      fake_alpha_operator.initialize_vector(alpha_solution);

      fake_a_operator.initialize_vector(psi_solution);
      fake_a_operator.initialize_vector(phi_solution);
      fake_a_operator.initialize_vector(pi_solution);
      // Here phi and psi are interpolated to their initial values
      VectorTools::interpolate(dof_handler_DG, phi_init<dim>(),phi_solution);
      VectorTools::interpolate(dof_handler_DG, psi_init<dim>(),psi_solution);

     dof_r_val.resize(dof_handler_DG.n_dofs());

     DoFTools::map_dofs_to_support_points(mapping,dof_handler_DG,dof_r_val);

     // a_filler() then does the integration outwards and backin
     a_filler();

     // This is just for convergence analysis and shouldn't be needed.
     if (initial_convergence_switch)
     {
       if (cycle > fine_cycle)
       {
         output_results_IC_special(cycle);
       }
       if (cycle ==  coarse_cycle)
       {
         fake_a_operator.initialize_vector(a_coarse_solution);
         fake_a_operator.initialize_vector(alpha_coarse_solution);
         fake_a_operator.initialize_vector(phi_coarse_solution);
         fake_a_operator.initialize_vector(psi_coarse_solution);
         a_coarse_solution = a_solution;
         alpha_coarse_solution = alpha_solution;
         phi_coarse_solution = phi_solution;
         psi_coarse_solution = psi_solution;
         dof_points_coarse = dof_r_val;
       }
       if (cycle ==  fine_cycle)
       {
         fake_a_operator.initialize_vector(a_fine_solution);
         fake_a_operator.initialize_vector(alpha_fine_solution);
         fake_a_operator.initialize_vector(phi_fine_solution);
         fake_a_operator.initialize_vector(psi_fine_solution);
         a_fine_solution = a_solution;
         alpha_fine_solution = alpha_solution;
         phi_fine_solution = phi_solution;
         psi_fine_solution = psi_solution;
         dof_points_fine = dof_r_val;
       }
     }
     if (output_prints)
     {
     output_results_IC(cycle);
     }
     cycle += 1;
   }

   // outputs the convergence results
   if (initial_convergence_switch)
   {
    initial_convergence_test();
   }
      pcout << std::endl;

    }
  }

  template <int dim>
  void EvolutionProblem<dim>::initial_convergence_test()
  {
    /*
    This function was just used as a quick and dirty way to confirm that intial
    solutions of a, alpha, phi, and psi were converging with the expected
    convergence rates. Not the most clever way but it works.
    */
    double a_diff_coarse = 0;
    double alpha_diff_coarse = 0;
    double phi_diff_coarse = 0;
    double psi_diff_coarse = 0;


    double a_diff_fine = 0;
    double alpha_diff_fine = 0;
    double phi_diff_fine = 0;
    double psi_diff_fine = 0;

    pow_func<dim> pow_f;

    const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler_DG);
    LinearAlgebra::distributed::Vector<double> a_coarse_point(a_solution);
    LinearAlgebra::distributed::Vector<double> a_fine_point(a_solution);
    LinearAlgebra::distributed::Vector<double> a_true_point(a_solution);
    LinearAlgebra::distributed::Vector<double> alpha_coarse_point(a_solution);
    LinearAlgebra::distributed::Vector<double> alpha_fine_point(a_solution);
    LinearAlgebra::distributed::Vector<double> alpha_true_point(a_solution);
    LinearAlgebra::distributed::Vector<double> phi_coarse_point(a_solution);
    LinearAlgebra::distributed::Vector<double> phi_fine_point(a_solution);
    LinearAlgebra::distributed::Vector<double> psi_coarse_point(a_solution);
    LinearAlgebra::distributed::Vector<double> psi_fine_point(a_solution);
    a_coarse_point.reinit(dof_handler_DG.locally_owned_dofs(),
                    locally_relevant_dofs,
                    triangulation.get_communicator());
    a_coarse_point.copy_locally_owned_data_from(a_coarse_solution);
    constraints_DG.distribute(a_coarse_point);
    a_fine_point.reinit(dof_handler_DG.locally_owned_dofs(),
                    locally_relevant_dofs,
                    triangulation.get_communicator());
    a_fine_point.copy_locally_owned_data_from(a_fine_solution);
    constraints_DG.distribute(a_fine_point);
    a_true_point.reinit(dof_handler_DG.locally_owned_dofs(),
                    locally_relevant_dofs,
                    triangulation.get_communicator());
    a_true_point.copy_locally_owned_data_from(a_solution);
    constraints_DG.distribute(a_true_point);

    alpha_coarse_point.reinit(dof_handler_DG.locally_owned_dofs(),
                    locally_relevant_dofs,
                    triangulation.get_communicator());
    alpha_coarse_point.copy_locally_owned_data_from(alpha_coarse_solution);
    constraints_DG.distribute(alpha_coarse_point);
    alpha_fine_point.reinit(dof_handler_DG.locally_owned_dofs(),
                    locally_relevant_dofs,
                    triangulation.get_communicator());
    alpha_fine_point.copy_locally_owned_data_from(alpha_fine_solution);
    constraints_DG.distribute(alpha_fine_point);
    alpha_true_point.reinit(dof_handler_DG.locally_owned_dofs(),
                    locally_relevant_dofs,
                    triangulation.get_communicator());
    alpha_true_point.copy_locally_owned_data_from(alpha_solution);
    constraints_DG.distribute(alpha_true_point);

    phi_coarse_point.reinit(dof_handler_DG.locally_owned_dofs(),
                    locally_relevant_dofs,
                    triangulation.get_communicator());
    phi_coarse_point.copy_locally_owned_data_from(phi_coarse_solution);
    constraints_DG.distribute(phi_coarse_point);
    phi_fine_point.reinit(dof_handler_DG.locally_owned_dofs(),
                    locally_relevant_dofs,
                    triangulation.get_communicator());
    phi_fine_point.copy_locally_owned_data_from(phi_fine_solution);
    constraints_DG.distribute(phi_fine_point);

    psi_coarse_point.reinit(dof_handler_DG.locally_owned_dofs(),
                    locally_relevant_dofs,
                    triangulation.get_communicator());
    psi_coarse_point.copy_locally_owned_data_from(psi_coarse_solution);
    constraints_DG.distribute(psi_coarse_point);
    psi_fine_point.reinit(dof_handler_DG.locally_owned_dofs(),
                    locally_relevant_dofs,
                    triangulation.get_communicator());
    psi_fine_point.copy_locally_owned_data_from(psi_fine_solution);
    constraints_DG.distribute(psi_fine_point);

    a_coarse_point.update_ghost_values();
    a_fine_point.update_ghost_values();
    a_true_point.update_ghost_values();
    alpha_coarse_point.update_ghost_values();
    alpha_fine_point.update_ghost_values();
    alpha_true_point.update_ghost_values();
    phi_coarse_point.update_ghost_values();
    phi_fine_point.update_ghost_values();
    psi_coarse_point.update_ghost_values();
    psi_fine_point.update_ghost_values();

    const QGauss<dim> quadrature_formula(fe_degree + 2);

    FEValues<dim>     fe_values(fe_DG,
                            quadrature_formula,
                            update_values |   update_quadrature_points |
                              update_JxW_values);



    const unsigned int dofs_per_cell = fe_DG.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double>  a_coarse_values(n_q_points);
    std::vector<double> a_fine_values(n_q_points);
    std::vector<double> a_true_values(n_q_points);
    std::vector<double> alpha_coarse_values(n_q_points);
    std::vector<double> alpha_fine_values(n_q_points);
    std::vector<double> alpha_true_values(n_q_points);
    std::vector<double> phi_coarse_values(n_q_points);
    std::vector<double> phi_fine_values(n_q_points);
    std::vector<double> psi_coarse_values(n_q_points);
    std::vector<double> psi_fine_values(n_q_points);

    phi_init<dim> phi_f;
    psi_init<dim> psi_f;

    for (const auto &cell : dof_handler_DG.active_cell_iterators())
      {
        fe_values.reinit(cell);


        fe_values.get_function_values(a_coarse_point,a_coarse_values);
        fe_values.get_function_values(a_fine_point,a_fine_values);
        fe_values.get_function_values(a_true_point, a_true_values);
        fe_values.get_function_values(alpha_coarse_point,alpha_coarse_values);
        fe_values.get_function_values(alpha_fine_point,alpha_fine_values);
        fe_values.get_function_values(alpha_true_point, alpha_true_values);

        fe_values.get_function_values(phi_coarse_point,phi_coarse_values);
        fe_values.get_function_values(phi_fine_point,phi_fine_values);
        fe_values.get_function_values(psi_coarse_point,psi_coarse_values);
        fe_values.get_function_values(psi_fine_point,psi_fine_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            a_diff_coarse += pow_f.eval(a_true_values[q] - a_coarse_values[q],2)*fe_values.JxW(q);
            a_diff_fine   += pow_f.eval(a_true_values[q] - a_fine_values[q],2)*fe_values.JxW(q);
            alpha_diff_coarse += pow_f.eval(alpha_true_values[q] - alpha_coarse_values[q],2)*fe_values.JxW(q);
            alpha_diff_fine   += pow_f.eval(alpha_true_values[q] - alpha_fine_values[q],2)*fe_values.JxW(q);
            phi_diff_coarse += pow_f.eval(phi_f.value(fe_values.quadrature_point(q)) - phi_coarse_values[q],2)*fe_values.JxW(q);
            phi_diff_fine   += pow_f.eval(phi_f.value(fe_values.quadrature_point(q)) - phi_fine_values[q],2)*fe_values.JxW(q);
            psi_diff_coarse += pow_f.eval(psi_f.value(fe_values.quadrature_point(q)) - psi_coarse_values[q],2)*fe_values.JxW(q);
            psi_diff_fine   += pow_f.eval(psi_f.value(fe_values.quadrature_point(q)) - psi_fine_values[q],2)*fe_values.JxW(q);
            }

    }
    a_diff_coarse = sqrt(a_diff_coarse);
    alpha_diff_coarse = sqrt(alpha_diff_coarse);
    phi_diff_coarse = sqrt(phi_diff_coarse);
    psi_diff_coarse = sqrt(psi_diff_coarse);

    a_diff_fine = sqrt(a_diff_fine);
    alpha_diff_fine = sqrt(alpha_diff_fine);
    phi_diff_fine = sqrt(phi_diff_fine);
    psi_diff_fine = sqrt(psi_diff_fine);

    std::cout << "Convergence ratio for a: " << a_diff_coarse/a_diff_fine << std::endl;
    std::cout << "Convergence ratio for alpha: " << alpha_diff_coarse/alpha_diff_fine << std::endl;
    std::cout << "Convergence ratio for phi: " << phi_diff_coarse/phi_diff_fine << std::endl;
    std::cout << "Convergence ratio for psi: " << psi_diff_coarse/psi_diff_fine << std::endl;
  }



   template <int dim>
   void EvolutionProblem<dim>::IC_refinement(double &error_estimate, unsigned int &cycle)
   {
     //The function in charge of flagging cells for refinement and then executing
     //the refinement after each solution of initial conditions.
     //Bases cell error off of strong from of a outwards eqn violation then refines
     //either adaptively or uniformly depending on flags set at top of file.
     //Also, transfers solutions from unrefined mesh to the more refined mesh.

     // First part just fills the vector constraint_violation with the error
     // of each cell in the same order as the active_cell_iterator()
     constraint_violation.reinit(0);
     constraint_violation.reinit(triangulation.n_active_cells());

     const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler_DG);
     LinearAlgebra::distributed::Vector<double> psi_point(psi_solution);
     LinearAlgebra::distributed::Vector<double> pi_point(pi_solution);
     LinearAlgebra::distributed::Vector<double> a_point(a_solution);
     psi_point.reinit(dof_handler_DG.locally_owned_dofs(),
                     locally_relevant_dofs,
                     triangulation.get_communicator());
     psi_point.copy_locally_owned_data_from(psi_solution);
     constraints_DG.distribute(psi_point);
     pi_point.reinit(dof_handler_DG.locally_owned_dofs(),
                     locally_relevant_dofs,
                     triangulation.get_communicator());
     pi_point.copy_locally_owned_data_from(pi_solution);
     constraints_DG.distribute(pi_point);
     a_point.reinit(dof_handler_DG.locally_owned_dofs(),
                     locally_relevant_dofs,
                     triangulation.get_communicator());
     a_point.copy_locally_owned_data_from(a_solution);
     constraints_DG.distribute(a_point);

     psi_point.update_ghost_values();
     pi_point.update_ghost_values();
     a_point.update_ghost_values();
     const QGauss<dim> quadrature_formula(fe_degree + 2);

     FEValues<dim>     fe_values(fe_DG,
                             quadrature_formula,
                             update_values |  update_gradients | update_quadrature_points |
                               update_JxW_values);



     const unsigned int dofs_per_cell = fe_DG.n_dofs_per_cell();
     const unsigned int n_q_points    = quadrature_formula.size();

     std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

     std::vector<double>  a_values(n_q_points);
     std::vector<Tensor<1, dim>>  a_grad(n_q_points);
     std::vector<double> psi_values(n_q_points);
     std::vector<double> pi_values(n_q_points);

     int iterator_count=0;

     pow_func<dim> pow_f;

     for (const auto &cell : dof_handler_DG.active_cell_iterators())
       {
         fe_values.reinit(cell);


         fe_values.get_function_values(a_point,a_values);
         fe_values.get_function_gradients(a_point,a_grad);
         fe_values.get_function_values(psi_point,psi_values);
         fe_values.get_function_values(pi_point, pi_values);

          // Here is where the actual equation for the strong form of the a outwards
          // is input. It is squared and then multiplied by r^2 and the guassiang weight
          // Then after the loop for the cell is done then we take the sqrt of the sum.
         for (unsigned int q = 0; q < n_q_points; ++q)
           {
             constraint_violation(iterator_count) += (a_grad[q][0]*pow_f.eval(a_values[q],-1)  //dr_a/a
             - (1.-a_values[q]*a_values[q])*pow_f.eval(2.*fe_values.quadrature_point(q)[0],-1) // - (1-a^2)/r^2
             - 2.*numbers::PI*fe_values.quadrature_point(q)[0]*(                               // - 2*pi*r*(psi^2 + pi^2)   
                                                      psi_values[q]*psi_values[q]
                                                      + pi_values[q]*pi_values[q]
                                                    ))

                             *(a_grad[q][0]*pow_f.eval(a_values[q],-1)
                             - (1.-a_values[q]*a_values[q])*pow_f.eval(2.*fe_values.quadrature_point(q)[0],-1)
                             - 2.*numbers::PI*fe_values.quadrature_point(q)[0]*(
                                                                      psi_values[q]*psi_values[q]
                                                                      + pi_values[q]*pi_values[q]
                                                                    ))   // squared
                             * fe_values.quadrature_point(q)(0) * fe_values.quadrature_point(q)(0) * fe_values.JxW(q);             // rho*rho * dx
           }
           constraint_violation(iterator_count) = sqrt(constraint_violation(iterator_count));
           iterator_count += 1;
   }
   error_estimate = constraint_violation.linfty_norm();
   //std::cout << "Error_estimate: " << error_estimate << std::endl;

  // Now is the actual refinement part. Cells that have an error larger than half of the individual cell error limit
  // are flagged for refinement. If the cell error is less than 1/40th of the individual cell error limit then the cell
  // is flagged for coarsening. The prepare coarsening and refinement takes care of making sure
  // cells are only separated by one level of refinement and removes and changes flags accordingly.

  // For adaptive refinement no solution transfer is needed as the solutions are just recalculated from scratch.
  // For uniform refinement if the convergence analysis is being done the solution transfer 
  // just helps bring coarser representations to the finer mesher for easier comparison.
   if (!uniform_refinement && error_estimate > IC_individual_cell_error_limit){


     GridRefinement::refine(triangulation,constraint_violation,IC_individual_cell_error_limit/2);
     GridRefinement::coarsen(triangulation,constraint_violation,IC_individual_cell_error_limit/40);
     triangulation.prepare_coarsening_and_refinement();
     triangulation.execute_coarsening_and_refinement();
     dof_handler_DG.distribute_dofs(fe_DG);
     make_constraints();
   }
   else
   {
       SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>> soltrans(dof_handler_DG);
       soltrans.prepare_for_pure_refinement();
     triangulation.refine_global(1);
     dof_handler_DG.distribute_dofs(fe_DG);
     if (initial_convergence_switch)
     {
       if (cycle > coarse_cycle)
       {
         std::vector<LinearAlgebra::distributed::Vector<double>> solutions_vec = {a_coarse_solution, alpha_coarse_solution, phi_coarse_solution, psi_coarse_solution};
         a_coarse_solution.reinit(dof_handler_DG.n_dofs());
         alpha_coarse_solution.reinit(dof_handler_DG.n_dofs());
         phi_coarse_solution.reinit(dof_handler_DG.n_dofs());
         psi_coarse_solution.reinit(dof_handler_DG.n_dofs());
         soltrans.refine_interpolate(solutions_vec[0],a_coarse_solution);
         soltrans.refine_interpolate(solutions_vec[1],alpha_coarse_solution);
         soltrans.refine_interpolate(solutions_vec[2],phi_coarse_solution);
         soltrans.refine_interpolate(solutions_vec[3],psi_coarse_solution);
       }
       if (cycle > fine_cycle)
       {
         std::vector<LinearAlgebra::distributed::Vector<double>> solutions_vec = {a_fine_solution, alpha_fine_solution, phi_fine_solution, psi_fine_solution};
         a_fine_solution.reinit(dof_handler_DG.n_dofs());
         alpha_fine_solution.reinit(dof_handler_DG.n_dofs());
         phi_fine_solution.reinit(dof_handler_DG.n_dofs());
         psi_fine_solution.reinit(dof_handler_DG.n_dofs());
         soltrans.refine_interpolate(solutions_vec[0],a_fine_solution);
         soltrans.refine_interpolate(solutions_vec[1],alpha_fine_solution);
         soltrans.refine_interpolate(solutions_vec[2],phi_fine_solution);
         soltrans.refine_interpolate(solutions_vec[3],psi_fine_solution);
       }
     }

     make_constraints();
   }
   }

   template <int dim>
   void EvolutionProblem<dim>::a_filler()
   {
     //This is the function that integrates outwards for a and then back
     //inwards for alpha. Currently only used for the initial conditions and
     //after complete time steps. Intermediary steps are covered manually
     //within the integrator functions in time_a_alpha_integrators.h
     //Both use the fake_a_operator.h and fake_alpha_operator.h
     const LowStorageRungeKuttaIntegrator integrator(lsrk_scheme);

     double h_step_a;

     double current_position = 0;

     a_solution[0] = 1.;
     double a_val = 1.;
     double rk_a = 1.;
     for (unsigned int i=1; i<dof_handler_DG.n_dofs(); ++i){
         h_step_a = (dof_r_val[i][0] - current_position);
         while(abs(current_position - dof_r_val[i][0]) > 5e-16 && current_position < dof_r_val[i][0]){
           integrator.perform_integration_a(fake_a_operator,
                                          current_position,
                                          h_step_a,
                                          a_val,
                                          rk_a,
                                          fe_DG,
                                          i,
                                          dof_handler_DG);
          current_position += h_step_a;
          rk_a = a_val;
         }
         a_solution.local_element(i) = a_val;}

     double alpha_val = 1./a_val;
     alpha_solution(dof_handler_DG.n_dofs() - 1) = alpha_val;
     rk_a = alpha_val;

     for (unsigned int i=dof_handler_DG.n_dofs()-1; i>0; --i){
         h_step_a = (dof_r_val[i-1][0] - current_position);
         while(abs(current_position - dof_r_val[i-1][0]) > 5e-16 && current_position > dof_r_val[i-1][0]){
           integrator.perform_integration_a(fake_alpha_operator,
                                          current_position,
                                          h_step_a,
                                          alpha_val,
                                          rk_a,
                                          fe_DG,
                                          i,
                                          dof_handler_DG);
          current_position += h_step_a;
          rk_a = alpha_val;
         }
         alpha_solution[i-1] = alpha_val;
     }
   }

   template <int dim>
   void EvolutionProblem<dim>::Hamiltonian_Violation(double &error_estimate)
   {
     //This gives an estimate of error after every time step and was originally planned for
     //adaptive refinement flagging. It is done just with the strong form of the alpha integrating
     //inwards equation. Ignore the Hamiltonian name. It is an artifact of an older version.

     //It works very similar to the IC_refinement function. Iterates over all cells in the order
     //of the active_cell_iterator() and fills the constraint_violation vector with the error
     //of each cell. Then flags and refines and coarsens appropriately.
     constraint_violation.reinit(0);
     constraint_violation.reinit(triangulation.n_active_cells());
     //double mass = 0;

     const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler_DG);
     LinearAlgebra::distributed::Vector<double> psi_point(psi_solution);
     LinearAlgebra::distributed::Vector<double> pi_point(pi_solution);
     LinearAlgebra::distributed::Vector<double> a_point(a_solution);
     LinearAlgebra::distributed::Vector<double> alpha_point(alpha_solution);
     psi_point.reinit(dof_handler_DG.locally_owned_dofs(),
                     locally_relevant_dofs,
                     triangulation.get_communicator());
     psi_point.copy_locally_owned_data_from(psi_solution);
     constraints_DG.distribute(psi_point);
     pi_point.reinit(dof_handler_DG.locally_owned_dofs(),
                     locally_relevant_dofs,
                     triangulation.get_communicator());
     pi_point.copy_locally_owned_data_from(pi_solution);
     constraints_DG.distribute(pi_point);
     a_point.reinit(dof_handler_DG.locally_owned_dofs(),
                     locally_relevant_dofs,
                     triangulation.get_communicator());
     a_point.copy_locally_owned_data_from(a_solution);
     constraints_DG.distribute(a_point);
     alpha_point.reinit(dof_handler_DG.locally_owned_dofs(),
                     locally_relevant_dofs,
                     triangulation.get_communicator());
     alpha_point.copy_locally_owned_data_from(alpha_solution);
     constraints_DG.distribute(alpha_point);

     psi_point.update_ghost_values();
     pi_point.update_ghost_values();
     a_point.update_ghost_values();
     alpha_point.update_ghost_values();
     const QGauss<dim> quadrature_formula(fe_degree + 2);

     FEValues<dim>     fe_values(fe_DG,
                             quadrature_formula,
                             update_values |  update_gradients | update_quadrature_points |
                               update_JxW_values);



     const unsigned int dofs_per_cell = fe_DG.n_dofs_per_cell();
     const unsigned int n_q_points    = quadrature_formula.size();

     std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

     std::vector<double>  a_values(n_q_points);
     std::vector<Tensor<1, dim>>  a_grad(n_q_points);
     std::vector<double>  alpha_values(n_q_points);
     std::vector<Tensor<1, dim>>  alpha_grad(n_q_points);
     std::vector<double> psi_values(n_q_points);
     std::vector<double> pi_values(n_q_points);

     int iterator_count=0;

     pow_func<dim> pow_f;

     for (const auto &cell : dof_handler_DG.active_cell_iterators())
       {
        fe_values.reinit(cell);


        fe_values.get_function_values(a_point,a_values);
        fe_values.get_function_gradients(a_point,a_grad);
        fe_values.get_function_values(alpha_point,alpha_values);
        fe_values.get_function_gradients(alpha_point,alpha_grad);
        fe_values.get_function_values(psi_point,psi_values);
        fe_values.get_function_values(pi_point, pi_values);


        // Here the strong from of the alpha integrating inwards equation is calculated for the L2 norm.
        // It is squared and then multiplied by r^2 and the guassiang weight
        for (unsigned int q = 0; q < n_q_points; ++q)
           {
              {
                             constraint_violation(iterator_count) += (alpha_grad[q][0]*pow_f.eval(alpha_values[q],-1) //dr_alpha/alpha
                             - a_grad[q][0]*pow_f.eval(a_values[q],-1)                                                // -dr_a/a
                             + (1. - a_values[q]*a_values[q])*pow_f.eval(fe_values.quadrature_point(q)[0],-1))        // - (1-a^2)/r
                                             *(alpha_grad[q][0]*pow_f.eval(alpha_values[q],-1)
                                             - a_grad[q][0]*pow_f.eval(a_values[q],-1)
                                             + (1. - a_values[q]*a_values[q])*pow_f.eval(fe_values.quadrature_point(q)[0],-1))  // squared
                                             * fe_values.quadrature_point(q)(0) * fe_values.quadrature_point(q)(0) * fe_values.JxW(q);   // rho*rho * dx
              }
              
            // Remnant for cacluating the mass of the system.
            /*mass += (pow_f.eval(fe_values.quadrature_point(q)[0]*psi_values[q],2)*pow_f.eval(a_values[q],-2) +
                      pow_f.eval(fe_values.quadrature_point(q)[0]*pi_values[q],2)*pow_f.eval(a_values[q],-2))
                      *fe_values.JxW(q);*/
           }
           constraint_violation(iterator_count) = sqrt(constraint_violation(iterator_count));
           // Here the cell error is changed if the cell is already at maximum refinement level
           // This is just so the code doesn't refine endlessly near the critical point
           if (cell->level() >= max_refinement_level &&
                constraint_violation(iterator_count) > individual_cell_error_limit/3.)
                {
                  constraint_violation(iterator_count) = individual_cell_error_limit/10.;
                }
           iterator_count += 1;
   }
   error_estimate = constraint_violation.linfty_norm();
   /*std::cout << "Error_estimate: " << error_estimate <<
                "\nMass: " << 2.*numbers::PI*mass << std::endl;*/

  // Here is where the refinement is enacted. If the error is larger than half of the individual cell error limit
  // then the cell is flagged for refinement. If the error is less than 1/40th of the individual cell error limit
  // then the cell is flagged for coarsening. The prepare coarsening and refinement takes care of making sure
  // cells are only separated by one level of refinement and removes and changes flags accordingly.
  // In addition the solutions need to be transferred from the old mesh to the new mesh and 
  // the degrees of freedom need to be renumbered giong outwards from the origin.
  // This has not been included in uniform refinement yet but could easily be copied over in future implementation.
   if (!uniform_refinement && (error_estimate > individual_cell_error_limit)){

     Tensor<1,dim> renumbering_direction;
     renumbering_direction[0] = 1.;

     SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>> soltrans(dof_handler_DG);


     GridRefinement::refine(triangulation,constraint_violation,individual_cell_error_limit/2.);
     GridRefinement::coarsen(triangulation,constraint_violation,individual_cell_error_limit/40.);
     triangulation.prepare_coarsening_and_refinement();

     std::vector<LinearAlgebra::distributed::Vector<double>> old_solutions_vec = {old_phi_solution,old_pi_solution,old_psi_solution,old_a_solution,old_alpha_solution};
     soltrans.prepare_for_coarsening_and_refinement(old_solutions_vec);
     triangulation.execute_coarsening_and_refinement();

     dof_handler_DG.distribute_dofs(fe_DG);
     DoFRenumbering::downstream(dof_handler_DG, renumbering_direction);
     phi_solution.reinit(dof_handler_DG.n_dofs());
     psi_solution.reinit(dof_handler_DG.n_dofs());
     pi_solution.reinit(dof_handler_DG.n_dofs());
     a_solution.reinit(dof_handler_DG.n_dofs());
     alpha_solution.reinit(dof_handler_DG.n_dofs());
     std::vector<LinearAlgebra::distributed::Vector<double>> solutions_vec = {phi_solution,pi_solution,psi_solution,a_solution,alpha_solution};

     soltrans.interpolate(old_solutions_vec,solutions_vec);
     phi_solution = solutions_vec[0];
     pi_solution = solutions_vec[1];
     psi_solution = solutions_vec[2];
     a_solution = solutions_vec[3];
     alpha_solution = solutions_vec[4];
     make_constraints();
   }
   else if (error_estimate > individual_cell_error_limit){
     triangulation.refine_global(1);
    }
 }

 template <int dim>
 void EvolutionProblem<dim>::convergence_test(const double &dt)
 {
   /*This function just fills a vector with the violation of the spare
   evolution equation. Uses a centered finite difference fourth order
   approximation of dt_a and then right hand side values.
   Results in saving many extra vectors so preferred if it is turned when
   doing actual runs.*/
   convergence_violation.reinit(dof_handler_DG.n_dofs());
   for (unsigned int i=0; i<dof_handler_DG.n_dofs(); ++i){
     convergence_violation[i] =
        ((-a_solution[i] + 8.*old_a_solution[i] - 8.*old_3_a_solution[i] + old_4_a_solution[i])/(12.*dt)
         -4.*numbers::PI*dof_r_val[i][0]*old_2_alpha_solution[i]*old_2_psi_solution[i]*old_2_pi_solution[i]);
   }

   convergence_cell_violation.reinit(0);
     convergence_cell_violation.reinit(triangulation.n_active_cells());
     //double mass = 0;

     const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler_DG);
     LinearAlgebra::distributed::Vector<double> convergence_point(psi_solution);
     convergence_point.reinit(dof_handler_DG.locally_owned_dofs(),
                     locally_relevant_dofs,
                     triangulation.get_communicator());
     convergence_point.copy_locally_owned_data_from(convergence_violation);
     constraints_DG.distribute(convergence_point);
     

     convergence_point.update_ghost_values();
     const QGauss<dim> quadrature_formula(fe_degree + 2);

     FEValues<dim>     fe_values(fe_DG,
                             quadrature_formula,
                             update_values | update_quadrature_points |
                               update_JxW_values);



     const unsigned int dofs_per_cell = fe_DG.n_dofs_per_cell();
     const unsigned int n_q_points    = quadrature_formula.size();

     std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

     std::vector<double>  convergence_values(n_q_points);

     int iterator_count=0;

     for (const auto &cell : dof_handler_DG.active_cell_iterators())
       {
        fe_values.reinit(cell);


        fe_values.get_function_values(convergence_point,convergence_values);


        // Here the strong from of the alpha integrating inwards equation is calculated for the L2 norm.
        // It is squared and then multiplied by r^2 and the guassiang weight
        for (unsigned int q = 0; q < n_q_points; ++q)
           {
              {
                             convergence_cell_violation(iterator_count) += (convergence_values[q])        // - (1-a^2)/r
                                             *(convergence_values[q])  // squared
                                             * fe_values.quadrature_point(q)(0) * fe_values.quadrature_point(q)(0) * fe_values.JxW(q);   // rho*rho * dx
              }
              
            // Remnant for cacluating the mass of the system.
            /*mass += (pow_f.eval(fe_values.quadrature_point(q)[0]*psi_values[q],2)*pow_f.eval(a_values[q],-2) +
                      pow_f.eval(fe_values.quadrature_point(q)[0]*pi_values[q],2)*pow_f.eval(a_values[q],-2))
                      *fe_values.JxW(q);*/
           }
           convergence_cell_violation(iterator_count) = sqrt(constraint_violation(iterator_count));
           // Here the cell error is changed if the cell is already at maximum refinement level
           // This is just so the code doesn't refine endlessly near the critical point
           iterator_count += 1;
   }
 }

 template <int dim>
 void EvolutionProblem<dim>::stop_check(struct Diagnostics &diag, unsigned int &time_step_number)
 {
   /*This checks whether NaN values have appeared.
      Diagnostics are just needed so it can flush out the 
      required files for critical point searches using a
      separate python script.*/
  for (unsigned int i = 0; i < dof_handler_DG.n_dofs(); ++i)
  {
    if (a_solution[i] != a_solution[i])
    {
      std::cout << "Error at: " << dof_r_val[i][0] << "\nSpacing: "
      << time_step << "\nTime: " << time << "\na_solution\n" <<
      std::setprecision(15) << seed_amp << " Collapsed" << std::endl;
      diag.flush();
      AssertThrow(false, ExcMessage("Error in a_solution"));
    }
    else if (alpha_solution[i] != alpha_solution[i])
    {
      std::cout << "Error at: " << dof_r_val[i][0] << "\nSpacing: "
      << time_step << "\nTime: " << time << "\nalpha_solution\n" <<
      std::setprecision(15) << seed_amp << " Collapsed" << std::endl;
      diag.flush();
      output_results_Evolve(ceil(time_step_number/output_spacing));
      AssertThrow(false, ExcMessage("Error in alpha_solution"));
    }
    else if (pi_solution[i] != pi_solution[i])
    {
      std::cout << "Error at: " << dof_r_val[i][0] << "\nSpacing: "
      << time_step << "\nTime: " << time << "\npi_solution\n" <<
      std::setprecision(15) << seed_amp << " Collapsed" << std::endl;
      diag.flush();
      AssertThrow(false, ExcMessage("Error in pi_solution"));
    }
    else if (phi_solution[i] != phi_solution[i])
    {
      std::cout << "Error at: " << dof_r_val[i][0] << "\nSpacing: "
      << time_step << "\nTime: " << time << "\nphi_solution\n" <<
      std::setprecision(15) << seed_amp << " Collapsed" << std::endl;
      diag.flush();
      AssertThrow(false, ExcMessage("Error in phi_solution"));
    }
    else if (psi_solution[i] != psi_solution[i])
    {
      std::cout << "Error at: " << dof_r_val[i][0] << "\nSpacing: "
      << time_step << "\nTime: " << time << "\npsi_solution\n" <<
      std::setprecision(15) << seed_amp << " Collapsed" << std::endl;
      diag.flush();
      AssertThrow(false, ExcMessage("Error in psi_solution"));
    }
  }
 }

template <int dim>
double EvolutionProblem<dim>::MassFinder() {
    //Determine mass of black hole via Hawking mass. Basically, find max of metric rr coeff
    //and take half the radial distance. Units are that of the initial total mass
    //of the spacetime, but for critical analysis this isn't too important
    //since the relevant corrections introduced quickly become smaller than that of
    //the numerical accuracy probed.

    double mass = 0.0;
    int ma = 0;
    for(unsigned int i=0;i<dof_handler_DG.n_dofs();i++) {
        if(a_solution[i]>a_solution[ma]) {
            ma=i;
        }
    }
    mass = 0.5*dof_r_val[ma][0];
    return mass;
}


   template <int dim>
   void EvolutionProblem<dim>::run(struct Diagnostics &diag)
   {
    /* The main function that runs the evolution.
       Diagnostics is used purely for a more efficient way
       of writing out files for critical searches.
       
       This ties all previous functions together.
       First it makes an initial mesh. Spreads out the dofs and 
       solves the initial conditions refineing the mesh appropriately.

       Then it initializes all the vectors and calculates the time step size
       using spatial spacing and the CFL factor. Then it starts the time
       loop.
    */
    std::cout << std::setprecision(17) << std::fixed;
     {
       const unsigned int n_vect_number = VectorizedArray<Number>::size();
       const unsigned int n_vect_bits   = 8 * sizeof(Number) * n_vect_number;

       if (output_prints)
       {
       pcout << "Running with "
             << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
             << " MPI processes" << std::endl;
       pcout << "Vectorization over " << n_vect_number << ' '
             << (std::is_same<Number, double>::value ? "doubles" : "floats")
             << " = " << n_vect_bits << " bits ("
             << Utilities::System::get_current_vectorization_level() << ')'
             << std::endl;
       }
     }

     make_grid();

     IC_gen();

     make_constraints();

     const LowStorageRungeKuttaIntegrator integrator(lsrk_scheme);

    // Calculating the time step size
     double min_vertex_distance = std::numeric_limits<double>::max();
     for (const auto &cell : triangulation.active_cell_iterators())
       if (cell->is_locally_owned())
         min_vertex_distance =
           std::min(min_vertex_distance, cell->minimum_vertex_distance());
     min_vertex_distance =
       Utilities::MPI::min(min_vertex_distance, MPI_COMM_WORLD);

     time_step = cfl_factor*min_vertex_distance;

     if (output_prints)
     {
     std::locale s = pcout.get_stream().getloc();
     pcout.get_stream().imbue(std::locale(""));
     pcout << "Number of degrees of freedom: " << dof_handler_DG.n_dofs()
           << " ( = " << " [vars] x "
           << triangulation.n_global_active_cells() << " [cells] x "
           << Utilities::pow(fe_degree + 1, dim) << " [dofs/cell/var] )"
           << std::endl;
     pcout.get_stream().imbue(s);

     pcout << "Time step size: " << time_step
           << ", minimal h: " << min_vertex_distance
           << std::endl
           << std::endl;
     }

    // Initializing the operators and solution vectors
     const std::vector<const DoFHandler<dim> *> dof_handlers = {&dof_handler_DG};
     const std::vector<const AffineConstraints<double> *> constraints = {&constraints_DG};

     phi_operator.reinit(mapping,dof_handlers,constraints);
     pi_operator.reinit(mapping,dof_handlers,constraints);
     psi_operator.reinit(mapping,dof_handlers,constraints);
     a_operator.reinit(mapping,dof_handlers,constraints);
     fake_a_operator.reinit(mapping,dof_handler_DG,constraints_DG);
     fake_alpha_operator.reinit(mapping,dof_handler_DG,constraints_DG);

     psi_operator.initialize_vector(new_psi_solution);
     phi_operator.initialize_vector(new_phi_solution);
     pi_operator.initialize_vector(new_pi_solution);
     a_operator.initialize_vector(new_a_solution);
     a_operator.initialize_vector(other_a_solution);
     other_a_solution = a_solution;
     pi_operator.initialize_vector(pi_solution);


     LinearAlgebra::distributed::Vector<double> rk_register_1;
     LinearAlgebra::distributed::Vector<double> rk_register_2;
     LinearAlgebra::distributed::Vector<double> rk_register_3;
     LinearAlgebra::distributed::Vector<double> rk_register_4;
     LinearAlgebra::distributed::Vector<double> rk_register_5;
     phi_operator.initialize_vector(rk_register_1);
     phi_operator.initialize_vector(rk_register_2);
     phi_operator.initialize_vector(rk_register_3);
     phi_operator.initialize_vector(rk_register_4);
     phi_operator.initialize_vector(rk_register_5);

     phi_operator.initialize_vector(old_phi_solution);
     phi_operator.initialize_vector(old_psi_solution);
     phi_operator.initialize_vector(old_pi_solution);
     phi_operator.initialize_vector(old_a_solution);
     phi_operator.initialize_vector(old_alpha_solution);

     if (convergence_test_switch){
       old_2_a_solution.reinit(a_solution);
       old_3_a_solution.reinit(a_solution);
       old_4_a_solution.reinit(a_solution);
       old_2_alpha_solution.reinit(alpha_solution);
       old_2_pi_solution.reinit(pi_solution);
       old_2_psi_solution.reinit(psi_solution);
     }

     unsigned int timestep_number = 0;

     std::vector<LinearAlgebra::distributed::Vector<double>> rk_register_first = {rk_register_1,rk_register_2,rk_register_3, rk_register_5};
     std::vector<LinearAlgebra::distributed::Vector<double>> solutions_vec = {new_phi_solution,new_pi_solution,new_psi_solution,new_a_solution};

     constraint_violation.reinit(triangulation.n_active_cells());

     if (output_prints)
     {
      output_results_Evolve(0);
     }
      // For if filtering is wanted these need to be uncommented.

      /*dof_handler_modal.distribute_dofs(fe_modal);
      oe_operator.reinit(mapping,dof_handler_modal);
      oe_operator.initialize_vector(modal_solution);*/

      double current_error = 0;
      unsigned int last_refinement = 0;


      std::ofstream clear_file("phi_origin.txt", std::ios_base::out);
      clear_file.close();
      std::ofstream outfile("phi_origin.txt", std::ios_base::app);
      if (!outfile)
      {
        AssertThrow(false, ExcMessage("Could not open file for writing phi at origin."))
      }
      outfile << std::setprecision(17) << std::fixed;
      outfile << time << " " << phi_solution[0] << "\n";
    
      
      std::ofstream dof_v_time_file("dof_v_time.txt");
      if(!dof_v_time_file)
      {
        AssertThrow(false, ExcMessage("Could not open file for writing dof v time."))
      }
      dof_v_time_file << std::setprecision(17) << std::fixed;
      dof_v_time_file << dof_handler_DG.n_dofs() << " " << time << "\n";

     // Beginning of the time loop
     while (time < final_time - 1e-12 && timestep_number < temp_max_step)
       {
         ++timestep_number;
         ++ last_refinement;
         //Passing updated solutions backwards to prepare for finding the next step
         if(convergence_test_switch){
           old_4_a_solution = old_3_a_solution;
           old_3_a_solution = old_2_a_solution;
           old_2_a_solution = old_a_solution;
           old_2_alpha_solution = old_alpha_solution;
           old_2_psi_solution = old_psi_solution;
           old_2_pi_solution = old_pi_solution;
         }
         old_phi_solution = phi_solution;
         old_pi_solution = pi_solution;
         old_psi_solution = psi_solution;
         old_a_solution = a_solution;
         old_alpha_solution = alpha_solution;

         {
          if (output_prints)
          {
           std::cout << "Time step: " << timestep_number << std::endl;
          }
          //Preparing all the std::vectors to be passed to the integrator.
          //Also, some reinitializing is needed to account for if a
          //refinement has occurred.
           new_psi_solution = psi_solution;
           new_pi_solution = pi_solution;
           new_phi_solution = phi_solution;
           new_a_solution = other_a_solution;
           std::vector<LinearAlgebra::distributed::Vector<double>> solutions_vec = {new_phi_solution,new_pi_solution,new_psi_solution,new_a_solution};

           phi_operator.initialize_vector(rk_register_1);
           phi_operator.initialize_vector(rk_register_2);
           phi_operator.initialize_vector(rk_register_3);
           phi_operator.initialize_vector(rk_register_4);
           phi_operator.initialize_vector(rk_register_5);
           std::vector<LinearAlgebra::distributed::Vector<double>> rk_register_first = {rk_register_1,rk_register_2,rk_register_3,rk_register_5};

           //The integrator does all of the stages of the prescribed runge-kutta scheme
           // it is found in time_a_alpha_integrators.h
           integrator.perform_time_step(phi_operator,
                                        pi_operator,
                                        psi_operator,
                                        a_operator,
                                        fake_a_operator,
                                        fake_alpha_operator,
                                        time,
                                        time_step,
                                        solutions_vec,
                                        rk_register_first,
                                        rk_register_4,
                                        dof_handler_DG,
                                        fe_DG);
         }

         //This is where filtering is done. Commented out as it is not needed.
         //Its initializations is also commented out and needs a part
         //uncommented in the refinement if statement below if it is wanted.
         /*oe_operator.convert_modal(dof_handler_DG, psi_solution, dof_handler_modal);
         oe_operator.filter(dof_handler_modal,fe_modal,modal_solution,modal_solution);
         oe_operator.convert_nodal(dof_handler_DG, psi_solution, dof_handler_modal);

         oe_operator.convert_modal(dof_handler_DG, phi_solution, dof_handler_modal);
         oe_operator.filter(dof_handler_modal,fe_modal,modal_solution,modal_solution);
         oe_operator.convert_nodal(dof_handler_DG, phi_solution, dof_handler_modal);

         oe_operator.convert_modal(dof_handler_DG, pi_solution, dof_handler_modal);
         oe_operator.filter(dof_handler_modal,fe_modal,modal_solution,modal_solution);
         oe_operator.convert_nodal(dof_handler_DG, pi_solution, dof_handler_modal);*/

         a_filler();

         approx_alpha += alpha_solution[0];
         approx_alpha /= (integrator.n_stages()+1);
         if (output_prints)
         {
         std::cout << "Approx alpha: " << approx_alpha << std::endl;
         }
         if (approx_alpha < 0.){
            std::cout << time_step << "\nTime: " << time << "\nphi_solution\n" <<
            std::setprecision(15) << seed_amp << " Collapsed" << std::endl;
            diag.flush();
            AssertThrow(false, ExcMessage("Error alpha can't be negative"))
         }
         

         stop_check(diag, timestep_number);

         //time += time_step*approx_alpha;
         time += time_step*alpha_solution[0];
         if (phi_origin_analysis){
          outfile << time << " " << phi_solution[0] << "\n";
         }
         if (output_prints)
         {
          std::cout << "Time origin: " << time << std::endl;
         }


         if (convergence_test_switch && timestep_number >= convergence_test_number && last_refinement > 4)
         {
           convergence_test(time_step);
         }

         if (timestep_number % output_spacing == 0 && output_prints)
         {
           output_results_Evolve(timestep_number/output_spacing);
         }

         //refinement if statement. checks if the step forward has caused a large error.
         // if so then it undoes the step and refines and tries again.
         // it takes the next step regardless so it doesn't get stuck never taking
         // a step forward.
         if (timestep_number % error_checking_spacing == 0 && last_refinement > 1)
         {
           Hamiltonian_Violation(current_error);

           if(current_error > individual_cell_error_limit){
             last_refinement = 0;
             timestep_number -= 1;
             time -= time_step*approx_alpha;

             dof_r_val.resize(dof_handler_DG.n_dofs());
             other_a_solution = a_solution;
             DoFTools::map_dofs_to_support_points(mapping,dof_handler_DG,dof_r_val);
             rk_register_1.reinit(phi_solution);
             rk_register_2.reinit(psi_solution);
             rk_register_3.reinit(pi_solution);
             rk_register_4.reinit(phi_solution);
             std::vector<LinearAlgebra::distributed::Vector<double>> rk_register_first = {rk_register_1,rk_register_2,rk_register_3};
              //min_vertex_distance = std::numeric_limits<double>::max();
             for (const auto &cell : triangulation.active_cell_iterators())
               if (cell->is_locally_owned())
                 min_vertex_distance =
                   std::min(min_vertex_distance, cell->minimum_vertex_distance());
             min_vertex_distance =
               Utilities::MPI::min(min_vertex_distance, MPI_COMM_WORLD);

             time_step = cfl_factor*min_vertex_distance;
             make_dofs();

             //Needs to be uncommented if filterting is wanted.
             /*dof_handler_modal.distribute_dofs(fe_modal);
             oe_operator.reinit(mapping,dof_handler_modal);
             oe_operator.initialize_vector(modal_solution);*/
           }
         }
         int max_level = 0;
         if (dof_time_analysis){
         min_vertex_distance = std::numeric_limits<double>::max();
         for (const auto &cell : triangulation.cell_iterators()){
                 min_vertex_distance = std::min(min_vertex_distance, cell->minimum_vertex_distance());
                 max_level = std::max(max_level, cell->level());}
          dof_v_time_file << dof_handler_DG.n_dofs() << " " << time << " " << max_level << " " << min_vertex_distance << " " << alpha_solution[0] << "\n";
         }
         if(output_diagnostics){
          diag.push(time,MassFinder(),alpha_solution[0]);
         }
      }
      if (phi_origin_analysis){
        outfile.close();
      }
      if (dof_time_analysis){
        dof_v_time_file.close();
      }

      if (mesh_error_analysis){
        std::ofstream error_mesh_file("error_mesh.txt");
        if(!error_mesh_file)
        {
          AssertThrow(false, ExcMessage("Could not open file for writing error mesh."))
        }
        error_mesh_file << std::setprecision(17) << std::fixed;

        // Write vec1 as comma-separated
          for (unsigned int i = 0; i < convergence_violation.size(); ++i)
          {
              error_mesh_file << convergence_violation[i];
              if (i != convergence_violation.size() - 1) error_mesh_file << ", ";
          }
          error_mesh_file << std::endl;

          // Write vec2 as comma-separated
          for (unsigned int i = 0; i < dof_r_val.size(); ++i)
          {
              error_mesh_file << dof_r_val[i];
              if (i != dof_r_val.size() - 1) error_mesh_file << ", ";
          }
          error_mesh_file << std::endl;
          error_mesh_file.close();

          std::cout << "Vectors written to error_mesh.txt" << std::endl;
      }


      //This just prints to the terminal if the solution has dispersed or collapsed
      if (time > final_time - .001)
        std::cout << std::setprecision(15) << seed_amp << " Dispersed" << std::endl;
      else
        std::cout << std::setprecision(15) << seed_amp << " Collapsed" << std::endl;
  }


} // namespace Scalar_Evolution



  //Main function reads in the command line arguments and starts the solver
  int main(int argc, char **argv)
  {
  using namespace Scalar_Evolution;
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  try
    {
      deallog.depth_console(0);
      if (argc != 5)
      {
        std::cout << "Please use ./[object] [amp] [ser] [output diagnostics] [output info and vtu]" << std::endl;
        // amp is a double for the initial amplitude of the scalar field
        // ser is the serial number for outputting numbers. It is just needed for
        // our critical search python scripts.
        // output diagnostics is a boolean for whether diagnostics are output for 
        // the critical search
        // output info and vtu is a boolean for whether the vtu files are output
        // along with printing information to the terminal
        exit(1);
      }
      seed_amp = std::atof(argv[1]);
      serial = atoi(argv[2]);
      output_diagnostics = atoi(argv[3]);
      output_prints = atoi(argv[4]);
      struct Diagnostics diag;
      EvolutionProblem<dimension> evolution_problem;
      evolution_problem.run(diag);
      if (output_diagnostics)
      {
        diag.flush();
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
  }
