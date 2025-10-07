//Operator for filtering and applying a limiter to the code
//Is not needed in the 1D code and really is not in a great shape
//right now. Pieces are there and this is just left
//in case I ever want to use it s parts again.

//Started with the idea of applying an oscillation eliminating
//scheme from another paper. All the parts are there it just isn't
//needed so it is left in a state of disrepair. Needs a lot of comments
//sorry hopefully it isn't too confusing if you ever use this.

namespace Scalar_Evolution
{
  using namespace dealii;


  template <int dim, int degree, int n_points_1d>
  class OE_Operator
  {
  public:
    static constexpr unsigned int n_quadrature_points_1d = n_points_1d;

    OE_Operator();

    void reinit(const Mapping<dim> &   mapping,
    const DoFHandler<dim>  &dof_handlers);

    void apply(
      const double                                     &tau,
      const double                                     &u_avg,
      const DoFHandler<dim>                            &dof_handler,
      const Mapping<dim>                               &mapping,
      const FE_DGP<dim>                                &fe,
      const LinearAlgebra::distributed::Vector<Number> &src,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const bool                                       &even) const;

    void filter(
      const DoFHandler<dim>                            &dof_handler,
      const FE_DGP<dim>                                &fe,
      const LinearAlgebra::distributed::Vector<Number> &src,
      LinearAlgebra::distributed::Vector<Number> &      dst) const;

    void limiter(
      const DoFHandler<dim>                            &dof_handler,
      const Mapping<dim>                               &mapping,
      const FESystem<dim>                                &fe,
      const LinearAlgebra::distributed::Vector<Number> &src,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const bool                                     &even) const;

    void convert_modal(
      const DoFHandler<dim>                            &dof_DG,
      const LinearAlgebra::distributed::Vector<Number> &sol,
      const DoFHandler<dim>                            &dof_modal) const;

    void convert_nodal(
      const DoFHandler<dim>                            &dof_DG,
      LinearAlgebra::distributed::Vector<Number>       &sol,
      const DoFHandler<dim>                            &dof_modal) const;

    void dissipation(
      const LinearAlgebra::distributed::Vector<Number>      &src,
      LinearAlgebra::distributed::Vector<Number>            &dst
    )const;

    void
    initialize_vector(LinearAlgebra::distributed::Vector<Number> &vector) const;

    MatrixFree<dim, Number> data;
  private:

    AffineConstraints<double> constraints_mod;

    void left_jump_filler(
      const FE_DGP<dim>                                &fe,
      const std::vector<Number>                        &cell,
      const std::vector<Number>                        &neighbor,
      std::vector<Number>                              &left_jump) const;

    void right_jump_filler(
      const FE_DGP<dim>                                &fe,
      const std::vector<Number>                        &cell,
      const std::vector<Number>                        &neighbor,
      std::vector<Number>                              &right_jump) const;

    double minmod(
      const double                                     &a_1,
      const double                                     &a_2,
      const double                                     &a_3) const;

    double cell_mean_(
      const std::vector<Number>                        &cell) const;
  };



  template <int dim, int degree, int n_points_1d>
  OE_Operator<dim, degree, n_points_1d>::OE_Operator()
  {}


    template <int dim, int degree, int n_points_1d>
    void OE_Operator<dim,degree,n_points_1d>::dissipation(
      const LinearAlgebra::distributed::Vector<Number>   &src,
      LinearAlgebra::distributed::Vector<Number>         &dst
    ) const
    {
      for(unsigned int i=0; i<src.size()-4; i++){
        if(i==2){
          dst[i] += 1/pow(2,6)*(src[i-2]-6*src[i-2]+15*src[i-1]-20*src[i]+15*src[i+1]-6*src[i+2]+src[i+3]);
        }
        else if(i==1){
          dst[i] += 1/pow(2,6)*(src[i]-6*src[i-1]+15*src[i-1]-20*src[i]+15*src[i+1]-6*src[i+2]+src[i+3]);
        }
        else if (i==0){
          dst[i] += 1/pow(2,6)*(src[i+2]-6*src[i+1]+15*src[i]-20*src[i]+15*src[i+1]-6*src[i+2]+src[i+3]);
        }
        else{
          dst[i] += 1/pow(2,6)*(src[i-3]-6*src[i-2]+15*src[i-1]-20*src[i]+15*src[i+1]-6*src[i+2]+src[i+3]);
        }
      }
    }

  // construction.
  template <int dim, int degree, int n_points_1d>
  void OE_Operator<dim, degree, n_points_1d>::reinit(
    const Mapping<dim> &   mapping,
    const DoFHandler<dim> &dof_handler)
  {

    const IndexSet locally_relevant_dofs_DG =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    constraints_mod.clear();
    constraints_mod.reinit(locally_relevant_dofs_DG);
    constraints_mod.close();

    const std::vector<const DoFHandler<dim> *> dof_handlers = {&dof_handler};
    const std::vector<const AffineConstraints<double> *> constraints = {&constraints_mod};

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
  void OE_Operator<dim, degree, n_points_1d>::initialize_vector(
    LinearAlgebra::distributed::Vector<Number> &vector) const
  {
    data.initialize_dof_vector(vector,0);
  }

  template <int dim, int degree, int n_points_1d>
  void OE_Operator<dim, degree, n_points_1d>::convert_modal(
    const DoFHandler<dim>                            &dof_DG,
    const LinearAlgebra::distributed::Vector<Number> &sol,
    const DoFHandler<dim>                            &dof_modal) const
  {
    FETools::project_dg(dof_DG, sol, dof_modal, modal_solution);
  }

  template <int dim, int degree, int n_points_1d>
  void OE_Operator<dim, degree, n_points_1d>::convert_nodal(
    const DoFHandler<dim>                            &dof_DG,
    LinearAlgebra::distributed::Vector<Number>       &sol,
    const DoFHandler<dim>                            &dof_modal) const
  {
    FETools::project_dg(dof_modal, modal_solution, dof_DG, sol);
  }


  template <int dim, int degree, int n_points_1d>
  void OE_Operator<dim, degree, n_points_1d>::apply(
    const double                                     &tau,
    const double                                     &u_avg,
    const DoFHandler<dim>                            &dof_handler,
    const Mapping<dim>                               &mapping,
    const FE_DGP<dim>                                &fe,
    const LinearAlgebra::distributed::Vector<Number> &src,
    LinearAlgebra::distributed::Vector<Number> &      /*dst*/,
    const bool                                       &even) const
  {
    //std::cout << tau << std::endl;
    LinearAlgebra::distributed::Vector<double> u_avg_vec;
    LinearAlgebra::distributed::Vector<double> sigma_vec;
    u_avg_vec.reinit(src);
    sigma_vec.reinit(src);
    sigma_vec = 0.;
    VectorTools::project(mapping,dof_handler,constraints_mod,QGauss<dim>(alt_q_points),Functions::ConstantFunction<dim>(u_avg),u_avg_vec);
    u_avg_vec -= src;
    double avg_norm = (u_avg_vec).linfty_norm();
    //std::cout << avg_norm << std::endl;
    //Done using u_avg_vec so going to reuse for the h_vec
    if (avg_norm > 0){
      std::vector<Number> cell_coeff(fe.dofs_per_cell);
      std::vector<Number> neighbor_coeff(fe.dofs_per_cell);
      std::vector<Number> left_jump(fe.degree + 1);
      std::vector<Number> right_jump(fe.degree + 1);
      //std::cout << "Made it before the cell iterator" << std::endl;
      for (const auto &cell: dof_handler.active_cell_iterators()){
        double sigma = 0.;
        const double h = cell->diameter();
        std::vector<unsigned int> local_dof_indices_cell(fe.dofs_per_cell);
        std::vector<unsigned int> local_dof_indices_neighbor(fe.dofs_per_cell);
        //std::cout << "Made it into the cell iterator" << std::endl;
        for (const auto &f : cell->face_indices()){
          //std::cout << "Made it into the face iterator" << cell->at_boundary(f) << std::endl;
          if (cell->at_boundary(f)){
            cell->get_dof_indices(local_dof_indices_cell);
            for (unsigned int i=0; i<fe.dofs_per_cell; ++i){
              cell_coeff[i] = src[local_dof_indices_cell[i]];
              //neighbor_coeff[i] = src[local_dof_indices_cell[i]];
              if (/*even ||*/ f==1){
                neighbor_coeff[i] = cell_coeff[i];
                //left_jump[i] = 0;
              }
              else{
                //neighbor_coeff[i] = -1.*cell_coeff[i];
                left_jump[i] = 0.;
              }
            }
            if (f == 0){
              //left_jump_filler(fe,cell_coeff,neighbor_coeff,left_jump);
              continue;
            }
            else if (f == 1){
              right_jump_filler(fe,cell_coeff,neighbor_coeff,right_jump);
            }
            else {
              AssertThrow(false, ExcMessage("This OEDG Method is only built for 1d"
                                            "extra faces were encountered."))
            }
          }
          else{
            auto neighbor = cell->neighbor(f);
            cell->get_dof_indices(local_dof_indices_cell);
            neighbor->get_dof_indices(local_dof_indices_neighbor);
            for (unsigned int i=0; i<fe.dofs_per_cell; ++i){
              cell_coeff[i] = src[local_dof_indices_cell[i]];
              neighbor_coeff[i] = src[local_dof_indices_neighbor[i]];
            }
            if (f == 0){
              left_jump_filler(fe,cell_coeff,neighbor_coeff,left_jump);
            }
            else if (f == 1){
              right_jump_filler(fe,cell_coeff,neighbor_coeff,right_jump);
            }
            else{
              AssertThrow(false, ExcMessage("This OEDG Method is only built for 1d"
                                            "extra faces were encountered."))
            }
          }
        }

        //Back in the cell loop
        for (unsigned int m = 0; m <= fe.degree; ++m){
          sigma += (2.*m + 1.)*std::pow(h,m)/((2.*fe.degree - 1.)*factorial(m))*
                    (std::abs(left_jump[m]) + std::abs(right_jump[m]))/(2.*avg_norm)*0.;
          if (m == 0){
            sigma_vec[local_dof_indices_cell[m]] = 0.;
            //std::cout << src[local_dof_indices_cell[m]] << std::endl;
          }
          else{
            sigma_vec[local_dof_indices_cell[m]] = sigma;
          }
          u_avg_vec[local_dof_indices_cell[m]] = h;
        }
      }
      //std::cout << *max_element(right_jump.begin(), right_jump.end()) << std::endl;
      //std::cout << *max_element(left_jump.begin(), left_jump.end()) << std::endl;
      //std::cout << sigma_vec.linfty_norm() << std::endl;
      for (unsigned int i = 0; i < src.size(); ++i){
        // u_avg_vec was reused to represent the size of mesh (ie. h)
        // I can change this around in the future for now it is a space saving idea.
        //std::cout << "Before: " << dst[i] << std::endl;
        //std::cout << std::exp(-tau/u_avg_vec[i]*sigma_vec[i]) << std::endl;
        if (std::exp(-tau/u_avg_vec[i]*sigma_vec[i]) < .6){
          //std::cout << std::exp(-tau/u_avg_vec[i]*sigma_vec[i]) << std::endl;
          //std::cout << i << std::endl;
        }
        modal_solution[i] *= std::exp(-tau/u_avg_vec[i]*sigma_vec[i]);
        //std::cout << "After: " << dst[i] << std::endl;
      }
    }
  }

  template <int dim, int degree, int n_points_1d>
  void OE_Operator<dim, degree, n_points_1d>::left_jump_filler(
    const FE_DGP<dim>                                &fe,
    const std::vector<Number>                        &cell,
    const std::vector<Number>                        &neighbor,
    std::vector<Number>                              &left_jump) const
  {
    Point<dim> cell_point(0);
    Point<dim> neighbor_point(1);
    if (fe.degree > 4){
      AssertThrow(false, ExcMessage("The Oscillation Eliminating filter is only"
                                    "built to allow up to 4th order basis funcitons."))
    }
    for (unsigned int m=0; m <= fe.degree; ++m){
      if (m==0){
        left_jump[m] = 0.;
        for (unsigned int mode = 0; mode <= fe.degree; ++mode){
          left_jump[m] += fe.shape_value(mode, cell_point)*cell[mode] - fe.shape_value(mode, neighbor_point)*neighbor[mode];
        }
      }
      else if (m==1){
        left_jump[m] = 0.;
        for (unsigned int mode = 0; mode <= fe.degree; ++mode){
          left_jump[m] += fe.shape_grad(mode, cell_point)[0]*cell[mode] - fe.shape_grad(mode, neighbor_point)[0]*neighbor[mode];
        }
      }
      else if (m==2){
        left_jump[m] = 0.;
        for (unsigned int mode = 0; mode <= fe.degree; ++mode){
          left_jump[m] += fe.shape_grad_grad(mode, cell_point)[0][0]*cell[mode] - fe.shape_grad_grad(mode, neighbor_point)[0][0]*neighbor[mode];
        }
      }
      else if (m==3){
        left_jump[m] = 0.;
        for (unsigned int mode = 0; mode <= fe.degree; ++mode){
          left_jump[m] += fe.shape_3rd_derivative(mode, cell_point)[0][0][0]*cell[mode] - fe.shape_3rd_derivative(mode, neighbor_point)[0][0][0]*neighbor[mode];
        }
      }
      else if (m==4){
        left_jump[m] = 0.;
        for (unsigned int mode = 0; mode <= fe.degree; ++mode){
          left_jump[m] += fe.shape_4th_derivative(mode, cell_point)[0][0][0][0]*cell[mode] - fe.shape_4th_derivative(mode, neighbor_point)[0][0][0][0]*neighbor[mode];
        }
      }
    }
  }

  template <int dim, int degree, int n_points_1d>
  void OE_Operator<dim, degree, n_points_1d>::right_jump_filler(
    const FE_DGP<dim>                                &fe,
    const std::vector<Number>                        &cell,
    const std::vector<Number>                        &neighbor,
    std::vector<Number>                              &right_jump) const
  {
    Point<dim> cell_point(1);
    Point<dim> neighbor_point(0);
    if (fe.degree > 4){
      AssertThrow(false, ExcMessage("The Oscillation Eliminating filter is only"
                                    "built to allow up to 4th order basis funcitons."))
    }
    for (unsigned int m=0; m <= fe.degree; ++m){
      if (m==0){
        right_jump[m] = 0.;
        for (unsigned int mode = 0; mode <= fe.degree; ++mode){
          right_jump[m] += fe.shape_value(mode, cell_point)*cell[mode] - fe.shape_value(mode, neighbor_point)*neighbor[mode];
        }
      }
      else if (m==1){
        right_jump[m] = 0.;
        for (unsigned int mode = 0; mode <= fe.degree; ++mode){
          right_jump[m] += fe.shape_grad(mode, cell_point)[0]*cell[mode] - fe.shape_grad(mode, neighbor_point)[0]*neighbor[mode];
        }
      }
      else if (m==2){
        right_jump[m] = 0.;
        for (unsigned int mode = 0; mode <= fe.degree; ++mode){
          right_jump[m] += fe.shape_grad_grad(mode, cell_point)[0][0]*cell[mode] - fe.shape_grad_grad(mode, neighbor_point)[0][0]*neighbor[mode];
        }
      }
      else if (m==3){
        right_jump[m] = 0.;
        for (unsigned int mode = 0; mode <= fe.degree; ++mode){
          right_jump[m] += fe.shape_3rd_derivative(mode, cell_point)[0][0][0]*cell[mode] - fe.shape_3rd_derivative(mode, neighbor_point)[0][0][0]*neighbor[mode];
        }
      }
      else if (m==4){
        right_jump[m] = 0.;
        for (unsigned int mode = 0; mode <= fe.degree; ++mode){
          right_jump[m] += fe.shape_4th_derivative(mode, cell_point)[0][0][0][0]*cell[mode] - fe.shape_4th_derivative(mode, neighbor_point)[0][0][0][0]*neighbor[mode];
        }
      }
    }
  }

  template <int dim, int degree, int n_points_1d>
  double OE_Operator<dim, degree, n_points_1d>::minmod(
    const double                                     &a_1,
    const double                                     &a_2,
    const double                                     &a_3) const
    {
      int s = sign_of(a_1) + sign_of(a_2) + sign_of(a_3);
      if (std::abs(s) == 3){
        return a_1/std::abs(a_1)*std::min({std::abs(a_1),std::abs(a_2),std::abs(a_3)});
      }
      else{
        return 0.;
      }
    }

    template <int dim, int degree, int n_points_1d>
    void OE_Operator<dim, degree, n_points_1d>::filter(
      const DoFHandler<dim>                            &dof_handler,
      const FE_DGP<dim>                                &fe,
      const LinearAlgebra::distributed::Vector<Number> &src,
      LinearAlgebra::distributed::Vector<Number> &      /*dst*/) const
      {
        LinearAlgebra::distributed::Vector<double> sigma_vec;
        sigma_vec.reinit(src);
        sigma_vec = 0.;
        const double N_ratio = 0./fe.dofs_per_cell;
        const int s          = 1;
        //std::vector<Number> cell_coeff(fe.dofs_per_cell);

        for (const auto &cell: dof_handler.active_cell_iterators()){

          std::vector<unsigned int> local_dof_indices_cell(fe.dofs_per_cell);
          cell->get_dof_indices(local_dof_indices_cell);

          for (unsigned int i=0; i<fe.dofs_per_cell; ++i){
            if (i >= N_ratio){
              sigma_vec[local_dof_indices_cell[i]] = std::exp(-36.*std::pow(((i*1.0/*+1.0*/)/(fe.dofs_per_cell/*+1.*/)-N_ratio)/(1.-N_ratio),2*s)*0.);
              //std::cout << local_dof_indices_cell[i] << std::endl;
              //std::cout << sigma_vec[local_dof_indices_cell[i]] << std::endl;
            }
            else{
              sigma_vec[local_dof_indices_cell[i]] = 1.;
              std::cout << "######################### \n I should not be in here in filter \n #####################" << std::endl;
            }
          }
        }
        for (unsigned int j=0; j<src.size(); ++j){
          modal_solution[j] *= sigma_vec[j];
        }
      }

    template <int dim, int degree, int n_points_1d>
    double OE_Operator<dim, degree, n_points_1d>::cell_mean_(
      const std::vector<Number>                        &cell) const
      {
        double cell_average = 0;
        for (unsigned int i = 0; i < cell.size(); ++i){
          cell_average += cell[i]/cell.size();  // This is essentially approximating integral by method of rectangles using the value at the left point.
          // No h size of the cell needed because they should be multiplied h/cell.size() then divide by h in the end. So tada it disappears.
        }
        return cell_average;
      }

  template <int dim, int degree, int n_points_1d>
  void OE_Operator<dim, degree, n_points_1d>::limiter(
    const DoFHandler<dim>                            &dof_handler,
    const Mapping<dim>                               &mapping,
    const FESystem<dim>                                &fe,
    const LinearAlgebra::distributed::Vector<Number> &src,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const bool                                     &even) const
    {
      //FEPointEvaluation<1,dim,dim,Number> alpha_test(mapping,fe,update_values|update_gradients|update_JxW_values|update_quadrature_points);
      /*const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
      LinearAlgebra::distributed::Vector<double> sol(src);
      sol.reinit(dof_handler_DG.locally_owned_dofs(),
                      locally_relevant_dofs,
                      triangulation.get_communicator());
      sol.copy_locally_owned_data_from(conformal_solution);

      sol.update_ghost_values();

      const QGauss<dim> quadrature_formula(n_points_1d);

      FEValues<dim>     fe_values(fe_DG,
                              quadrature_formula,
                              update_values | update_gradients | update_quadrature_points |
                                update_JxW_values);

      FEValues<dim>     fe_values_neighbor(fe_DG,
                              quadrature_formula,
                              update_values | update_quadrature_points |
                                update_JxW_values);



      const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

      std::vector<Tensor<1, dim>>  current_grad(n_points_1d);
      std::vector<double>  current_values(n_points_1d);
      std::vector<double>  neighbor_values(n_points_1d);*/

      std::vector<Number> cell_coeff(fe.dofs_per_cell);
      std::vector<Number> neighbor_coeff(fe.dofs_per_cell);
      std::vector<Point<dim>> supports;
      supports = fe.get_unit_support_points();
      std::vector<Point<dim>> coord(fe.dofs_per_cell);
      double cell_average;
      double left_average;
      double right_average;
      for (const auto &cell: dof_handler.active_cell_iterators()){
        const double h = cell->diameter();
        for (unsigned int q=0; q < supports.size(); ++q){
          coord[q] = mapping.transform_unit_to_real_cell(cell,supports[q]);
          //std::cout << supports[q][0] << std::endl;
          //std::cout << coord[q][0] << std::endl;
        }
        std::vector<unsigned int> local_dof_indices_cell(fe.dofs_per_cell);
        std::vector<unsigned int> local_dof_indices_neighbor(fe.dofs_per_cell);
        for (const auto &f : cell->face_indices()){

          if (cell->at_boundary(f)){
            cell->get_dof_indices(local_dof_indices_cell);
            for (unsigned int i=0; i<fe.dofs_per_cell; ++i){
              cell_coeff[i] = src[local_dof_indices_cell[i]];
              //neighbor_coeff[i] = src[local_dof_indices_cell[i]];
              if ((even && f == 0) || f==1){
                // This isn't quite right. Essentially instead of making the origin \/ it is making it //.
                // However, this isn't as much as problem when f == 1
                // Actually, deciding no changes needed since neighbor cell we only calculate the average and this shape difference should change nothing.
                neighbor_coeff[i] = cell_coeff[i];
              }
              else{
                neighbor_coeff[i] = -1.*cell_coeff[i];
              }
            }
            if (f == 0){
              //left_average = correct_val;
              left_average = cell_mean_(neighbor_coeff);
            }
            else if (f == 1){
              //right_average = correct_val;
              right_average = cell_mean_(neighbor_coeff);
            }
            else {
              AssertThrow(false, ExcMessage("This Limiter is only built for 1d"
                                            "extra faces were encountered."))
            }
          /*fe_values.reinit(cell);
          fe_values.get_function_values(sol,current_values);
          fe_values.get_function_gradients(sol,current_grad);*/
          }
          else{
            auto neighbor = cell->neighbor(f);
            //fe_values_neighbor.reinit(neighbor);
            cell->get_dof_indices(local_dof_indices_cell);
            neighbor->get_dof_indices(local_dof_indices_neighbor);
            for (unsigned int i=0; i<fe.dofs_per_cell; ++i){
              cell_coeff[i] = src[local_dof_indices_cell[i]];
              neighbor_coeff[i] = src[local_dof_indices_neighbor[i]];
            }
            if (f == 0){
              left_average = cell_mean_(neighbor_coeff);
            }
            else if (f == 1){
              right_average = cell_mean_(neighbor_coeff);
            }
            else{
              AssertThrow(false, ExcMessage("This OEDG Method is only built for 1d"
                                            "extra faces were encountered."))
            }
          }
        }
/* Crap for testing unit_gradient
        cell->get_dof_indices(local_dof_indices_cell);
        for (unsigned int i=0; i<fe.dofs_per_cell; ++i){
          cell_coeff[i] = alpha_solution[local_dof_indices_cell[i]];
        }

        alpha_test.reinit(cell,supports);
        alpha_test.evaluate(cell_coeff,EvaluationFlags::values | EvaluationFlags::gradients);
        for (unsigned int q = 0; q < cell_coeff.size(); ++q){
          std::cout << alpha_test.get_gradient(q) << std::endl;
          std::cout << "Unit Gradient: " << alpha_test.get_unit_gradient(q) << std::endl;
          std::cout << "Inverse Jacobian should be: " << alpha_test.get_gradient(q)[0]/alpha_test.get_unit_gradient(q)[0] << std::endl;
          //std::cout << alpha_test.inverse_jacobian(q) << std::endl;
          //std::cout << "Gradient should be: " << alpha_test.get_unit_gradient(q)*alpha_test.inverse_jacobian(q) << std::endl;
        }*/


        //Back in the cell loop
        cell_average = cell_mean_(cell_coeff);
        double cell_l = cell_coeff[0];
        double cell_r = cell_coeff[-1];
        double fake_slope = (cell_r - cell_l)/h;
        double v_l = cell_average - minmod(cell_average-cell_l,cell_average-left_average, right_average-cell_average);
        double v_r = cell_average + minmod(cell_r-cell_average,cell_average-left_average, right_average-cell_average);
        Point<dim> cell_center = cell->center();
        if (std::abs(v_l - cell_l) > 1e-19 || std::abs(v_r - cell_r) > 1e-19){
          for (unsigned int i=0; i<fe.dofs_per_cell; ++i){
            fake_slope = 0.;
            //std::cout << "Limiting cell centered at: " << cell_center[0] << std::endl
            for (unsigned int j=0; j<fe.dofs_per_cell; ++j){
              fake_slope += fe.shape_grad(j,supports[i])[0]*cell_coeff[j];
            }
            dst[local_dof_indices_cell[i]] = cell_average + (coord[i][0] - cell_center[0])*minmod(fake_slope,(right_average - cell_average)/(h/2.), (cell_average - left_average)/(h/2.));
          }
        }
      }
      /*for (unsigned int i = 0; i < src.size(); ++i){
      }*/
    }

} //namespace Brill_Evolution
