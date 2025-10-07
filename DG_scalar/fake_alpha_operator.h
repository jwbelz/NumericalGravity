//The operator for integrating alpha back inwards.

namespace Scalar_Evolution
{
  using namespace dealii;


  template <int dim, int degree, int n_points_1d>
  class fakeAlphaOperator
  {
  public:
    static constexpr unsigned int n_quadrature_points_1d = n_points_1d;

    fakeAlphaOperator();

    void reinit(const Mapping<dim> &   mapping,
    const DoFHandler<dim>  &dof_handlers,
    const AffineConstraints<double>  &constraints);

    void
    perform_stage(const Number current_position,
                  const Number factor_solution,
                  const Number h_,
                  const Number &current_ri,
                  Number &      solution,
                  Number &next_ri,
                  const FESystem<dim> &fe_DG,
                  const unsigned int &dof_num,
                  DoFHandler <dim>   &dof_handler) const;

    void
    initialize_vector(LinearAlgebra::distributed::Vector<Number> &vector) const;

    MatrixFree<dim, Number> data;

    void fill_shape_vals(const FESystem<dim> &fe_DG);
  private:

    std::array<double, 4*fe_degree*(fe_degree + 1)> DG_shape_vals;

    void local_apply_cell(
      const double &r,
      const double &alpha_val,
      double &dr,
      const FESystem<dim> &fe_DG,
      const unsigned int &cell_num,
      DoFHandler <dim> &dof_handler,
      const unsigned int &stage,
      const unsigned int &dof_interval) const;
  };



  template <int dim, int degree, int n_points_1d>
  fakeAlphaOperator<dim, degree, n_points_1d>::fakeAlphaOperator()
  {}


  template <int dim, int degree, int n_points_1d>
  void fakeAlphaOperator<dim, degree, n_points_1d>::reinit(
    const Mapping<dim> &   mapping,
    const DoFHandler<dim> &dof_handler_DG,
    const AffineConstraints<double>  &constraint_DG)
  {
    //Initializes the operator with knowledge of the degrees of freedom, coordinates, and constraints
    //Also informs what kind of information will be needed for integration
    const std::vector<const DoFHandler<dim> *> dof_handlers = {&dof_handler_DG};
    const std::vector<const AffineConstraints<double> *> constraints = {&constraint_DG};
    const std::vector<Quadrature<dim>> quadratures = {QGauss<dim>(alt_q_points),
                                                      QGauss<dim>(n_q_points_1d)};

    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points |
       update_values);
    additional_data.mapping_update_flags_inner_faces =
      (update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);
    additional_data.mapping_update_flags_boundary_faces =
      (update_JxW_values | update_quadrature_points | update_normal_vectors | update_gradients |
       update_values);
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim, Number>::AdditionalData::none;

    data.reinit(
      mapping, dof_handlers, constraints, quadratures, additional_data);
  }


  
  template <int dim, int degree, int n_points_1d>
  void fakeAlphaOperator<dim, degree, n_points_1d>::initialize_vector(
    LinearAlgebra::distributed::Vector<Number> &vector) const
  {
    //Initializes input vector so it is matched to the operator
    data.initialize_dof_vector(vector,0);
  }


  template <int dim, int degree, int n_points_1d>
  void fakeAlphaOperator<dim, degree, n_points_1d>::fill_shape_vals(
    const FESystem<dim> &fe_DG)
  {
    //This is for efficiency reasons. It fills what shape values will be for the basis
    //functions for the integrating inwards and outwards of the alpha equation.
    //This is facilitated by this operator being hardcoded to do RK4.

    //First finds the support points of the basis functions
    Point<dim> support_r_val;
    std::array<Point<dim>, 4*fe_degree> ref_r_val;
    double prev_point = 0.;
    for (unsigned int i=0; i<fe_degree; ++i)
    {
      //This stores the coordinates of the support point the (i+1)th basis function
      // on the reference unit cell.
      //First coordinate is 0 so we start at i+1.
      support_r_val = fe_DG.unit_support_point(i+1);
      //This stores all the coordinates that individual RK stages will be at for RK4.
      ref_r_val[i*(4)][0] = prev_point;
      ref_r_val[i*(4) + 1][0] = 0.5*(support_r_val[0] - prev_point) + prev_point;
      ref_r_val[i*(4) + 2][0] = ref_r_val[i*(4) + 1][0];
      ref_r_val[i*(4) + 3][0] = support_r_val[0];
      prev_point = support_r_val[0];
    }

    //This uses the coordinates of all the RK stages and then fills an array for what
    //all the DG basis function values will be at those points in order
    //so they can be used in the local_apply_cell function
    //to interpolate other functions to those points.
    for (unsigned int i=0; i<fe_degree; ++i)
    {
      for (unsigned int stage=0; stage < 4; ++stage)
      {
        for (unsigned int j=0; j<fe_degree+1; ++j)
        {
          DG_shape_vals[i*(4*(fe_degree+1)) + stage*(fe_degree + 1) + j] = fe_DG.shape_value(j, ref_r_val[i*(4) + stage]);
        }
      }
    }
  } 

  template <int dim, int degree, int n_points_1d>
  void fakeAlphaOperator<dim, degree, n_points_1d>::local_apply_cell(
    const double &r,
    const double &alpha_val,
    double &dr,
    const FESystem<dim> &fe_DG,
    const unsigned int &cell_num,
    DoFHandler <dim> &/*dof_handler*/,
    const unsigned int &stage,
    const unsigned int &dof_interval) const
  {
    //This calculates the dr_alpha for the RK4 integration of the alpha solution
    //Dof_handler is no longer used as it was for a super slow function
    //point_value() to interpolate values to the stage points.
    //It is left in the code commented out though.


    //Accounts for the NaN value at the origin that should be just 0
    if (r < 1e-12){
      dr = 0.;
    }
    // The main equation for solving for the integration of the alpha from
    // outer boundary inwards.
    else{
    psi_init<dim> psi_func;
      double a_val=0;
      //a_val = VectorTools::point_value(dof_handler,a_solution,r_val);
      double psi_val=0;
      //psi_val = VectorTools::point_value(dof_handler,psi_solution,r_val);
      double pi_val=0;
      //pi_val = VectorTools::point_value(dof_handler,pi_solution,r_val);

        //This for loop is the sum of the basis functions multiplied by their nodal values.
        //A much faster way for calculating the values of a,psi, and pi at the stage points.
        for (unsigned int i=0; i<fe_DG.n_dofs_per_cell(); ++i)
        {
          psi_val += DG_shape_vals[dof_interval*(4*(fe_degree + 1)) + (3-stage)*(fe_degree + 1) + i]*psi_solution[fe_DG.n_dofs_per_cell()*cell_num + i];
          pi_val += DG_shape_vals[dof_interval*(4*(fe_degree + 1)) + (3-stage)*(fe_degree + 1) + i]*pi_solution[fe_DG.n_dofs_per_cell()*cell_num + i];
          a_val += DG_shape_vals[dof_interval*(4*(fe_degree + 1)) + (3-stage)*(fe_degree + 1) + i]*a_solution[fe_DG.n_dofs_per_cell()*cell_num + i];
        }
        //Here we actually calculate dr_alpha for the RK4 integration
        dr = alpha_val*((1.-a_val*a_val)/(2.*r) + 2*numbers::PI*r*(psi_val*psi_val + pi_val*pi_val)
              + (a_val*a_val - 1.)/r);
    }
  }

  template <int dim, int degree, int n_points_1d>
  void fakeAlphaOperator<dim, degree, n_points_1d>::perform_stage(
    const Number                                      current_position,
    const Number                                      /*factor_solution*/,
    const Number                                      h_,
    const Number                                      &current_ri,
    Number &      solution,
    Number &      /*next_ri*/,
    const FESystem<dim>                               &fe_DG,
    const unsigned int                                &dof_num,
    DoFHandler <dim>                                  &dof_handler) const
  {
    //The driver of runge kutta solving the alpha equation
    double dr;
    double ki = 0.;
    //Data used just for optimization purposes.
    unsigned int cell_num = std::floor(dof_num/fe_DG.n_dofs_per_cell());
    unsigned int dof_interval = dof_num % fe_DG.n_dofs_per_cell() - 1;

      //The four stages of RK4
      local_apply_cell(current_position,current_ri,dr,fe_DG,cell_num,dof_handler,0,dof_interval);
      ki += dr;
      local_apply_cell(current_position+0.5*h_, current_ri+dr*0.5*h_,dr,fe_DG,cell_num,dof_handler,1,dof_interval);
      ki += 2.*dr;
      local_apply_cell(current_position+0.5*h_, current_ri+dr*0.5*h_,dr,fe_DG,cell_num,dof_handler,2,dof_interval);
      ki += 2.*dr;
      local_apply_cell(current_position+h_, current_ri + h_*dr, dr,fe_DG,cell_num,dof_handler,3,dof_interval);
      ki += dr;

      solution += h_/6.*ki;

  }

} //namespace Brill_Evolution
