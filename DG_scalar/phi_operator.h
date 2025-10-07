namespace Scalar_Evolution
{
  using namespace dealii;


  template <int dim, int degree, int n_points_1d>
  class Phi_Operator
  {
  public:
    static constexpr unsigned int n_quadrature_points_1d = n_points_1d;

    Phi_Operator();

    void reinit(const Mapping<dim> &   mapping,
    const std::vector<const DoFHandler<dim> *> &dof_handlers,
    const std::vector<const AffineConstraints<double> *> &constraints);

    void apply(const double                                      current_time,
               const LinearAlgebra::distributed::Vector<Number> &src,
               LinearAlgebra::distributed::Vector<Number> &      dst) const;

    void
    perform_stage(const Number cur_time,
                  const Number factor_solution,
                  const Number factor_ai,
                  const LinearAlgebra::distributed::Vector<Number> &current_ri,
                  LinearAlgebra::distributed::Vector<Number> &      vec_ki,
                  LinearAlgebra::distributed::Vector<Number> &      solution,
                  LinearAlgebra::distributed::Vector<Number> &next_ri) const;

    void
    initialize_vector(LinearAlgebra::distributed::Vector<Number> &vector) const;

    MatrixFree<dim, Number> data;
  private:

    void local_apply_inverse_mass_matrix(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;

    void local_apply_cell(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;

      pow_func<dim> pow_f;
  };



  template <int dim, int degree, int n_points_1d>
  Phi_Operator<dim, degree, n_points_1d>::Phi_Operator()
  {}

  template <int dim, int degree, int n_points_1d>
  void Phi_Operator<dim, degree, n_points_1d>::reinit(
    const Mapping<dim> &   mapping,
    const std::vector<const DoFHandler<dim> *> &dof_handlers,
    const std::vector<const AffineConstraints<double> *> &constraints)
  {/*
    This initializes the operator with knowledge of the mesh and
    the underlying degrees of freedom and constraints.
    Also, specifies what kind of information will be needed on the
    integration over the cells.
    */

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
  void Phi_Operator<dim, degree, n_points_1d>::initialize_vector(
    LinearAlgebra::distributed::Vector<Number> &vector) const
  {
    // Initializes the input vector so it is matched to the operator
    data.initialize_dof_vector(vector,0);
  }

  template <int dim, int degree, int n_points_1d>
  void Phi_Operator<dim, degree, n_points_1d>::local_apply_cell(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> & phi_vec,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    //This performs the integrations over cells of the rhs of phi evolution equation
    FEEvaluation<dim, degree, alt_q_points, 1, Number> phi(data,0);
    FEEvaluation<dim, degree, alt_q_points, 1, Number> alpha(data,0);
    FEEvaluation<dim, degree, alt_q_points, 1, Number> a(data,0);
    FEEvaluation<dim, degree, alt_q_points, 1, Number> pi(data,0);


    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(phi_vec, EvaluationFlags::values);
        alpha.reinit(cell);
        alpha.gather_evaluate(alpha_solution, EvaluationFlags::values);
        a.reinit(cell);
        a.gather_evaluate(a_solution, EvaluationFlags::values);
        pi.reinit(cell);
        pi.gather_evaluate(pi_solution, EvaluationFlags::values);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            phi.submit_value(alpha.get_value(q)*pi.get_value(q)*pow_f.eval(a.get_value(q),-1)
            ,q);

          }
        phi.integrate_scatter(EvaluationFlags::values,dst);
      }
  }

  template <int dim, int degree, int n_points_1d>
  void Phi_Operator<dim, degree, n_points_1d>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    //Quick calculation of the inverse mass matrix
    FEEvaluation<dim, degree, n_q_points_1d, 1, Number> psi(data, 0,1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, 1, Number>
      inverse(psi);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        psi.reinit(cell);
        psi.read_dof_values(src);
        inverse.apply(psi.begin_dof_values(), psi.begin_dof_values());

        psi.set_dof_values(dst);
      }
  }

  template <int dim, int degree, int n_points_1d>
  void Phi_Operator<dim, degree, n_points_1d>::apply(
    const double                                      current_time,
    const LinearAlgebra::distributed::Vector<Number> &src,
    LinearAlgebra::distributed::Vector<Number> &      dst) const
  {
    // Applies the operator to an input vector (src) and outputs result to dst
    {

      data.cell_loop(&Phi_Operator::local_apply_cell,
                this,
                dst,
                src,
                true);
    }

    {

      data.cell_loop(&Phi_Operator::local_apply_inverse_mass_matrix,
                     this,
                     dst,
                     dst);
    }
  }

  template <int dim, int degree, int n_points_1d>
  void Phi_Operator<dim, degree, n_points_1d>::perform_stage(
    const Number                                      /*current_time*/,
    const Number                                      factor_solution,
    const Number                                      factor_ai,
    const LinearAlgebra::distributed::Vector<Number> &current_ri,
    LinearAlgebra::distributed::Vector<Number> &      vec_ki,
    LinearAlgebra::distributed::Vector<Number> &      solution,
    LinearAlgebra::distributed::Vector<Number> &      next_ri) const
  {
    //This performs the stage of the RK scheme
    //First applies the operator to a vector.
    //Then applies the inverse mass matrix to the result of the first part
    //Then updates the solution vector
    {

      data.cell_loop(&Phi_Operator::local_apply_cell,
                this,
                vec_ki,
                current_ri,
                true);
    }

    {
      data.cell_loop(
        &Phi_Operator::local_apply_inverse_mass_matrix,
        this,
        next_ri,
        vec_ki,
        std::function<void(const unsigned int, const unsigned int)>(),
        [&](const unsigned int start_range, const unsigned int end_range) {
          const Number ai = factor_ai;
          const Number bi = factor_solution;
          if (ai == Number())
            {
              DEAL_II_OPENMP_SIMD_PRAGMA
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  const Number k_i          = next_ri.local_element(i);
                  const Number sol_i        = solution.local_element(i);
                  solution.local_element(i) = sol_i + bi * k_i;
                }
            }
          else
            {
              DEAL_II_OPENMP_SIMD_PRAGMA
              for (unsigned int i = start_range; i < end_range; ++i)
                {
                  const Number k_i          = next_ri.local_element(i);

                  const Number sol_i        = solution.local_element(i);
                  solution.local_element(i) = sol_i + bi * k_i;
                  next_ri.local_element(i)  = sol_i + ai * k_i;
                }
            }
        });
    }
  }

} //namespace Brill_Evolution
