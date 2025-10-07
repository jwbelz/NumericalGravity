namespace Scalar_Evolution
{
  using namespace dealii;


  template <int dim, int degree, int n_points_1d>
  class Pi_Operator
  {
  public:
    static constexpr unsigned int n_quadrature_points_1d = n_points_1d;

    Pi_Operator();

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

    void local_apply_face(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     face_range) const;

    void local_apply_boundary_face(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &     face_range) const;

      pow_func<dim> pow_f;
  };



  template <int dim, int degree, int n_points_1d>
  Pi_Operator<dim, degree, n_points_1d>::Pi_Operator()
  {}

  template <int dim, int degree, int n_points_1d>
  void Pi_Operator<dim, degree, n_points_1d>::reinit(
    const Mapping<dim> &   mapping,
    const std::vector<const DoFHandler<dim> *> &dof_handlers,
    const std::vector<const AffineConstraints<double> *> &constraints)
  {/*
    This initializes the pi operator with knowledge of the 
    degrees of freeedom, constraints, and underlying mesh.
    In addition, it tells what kind of information will
    need to be shared to the functions that integrate
    over the cell, faces, and boundaries.
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
      (update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim, Number>::AdditionalData::none;

    data.reinit(
      mapping, dof_handlers, constraints, quadratures, additional_data);
  }



  template <int dim, int degree, int n_points_1d>
  void Pi_Operator<dim, degree, n_points_1d>::initialize_vector(
    LinearAlgebra::distributed::Vector<Number> &vector) const
  {
    //Initializes the input vector so it is matched to the operator
    data.initialize_dof_vector(vector,0);
  }

  template <int dim, int degree, int n_points_1d>
  void Pi_Operator<dim, degree, n_points_1d>::local_apply_cell(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> & pi_vec,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    //This performs the integrations over cells of the rhs of pi evolution equation
    FEEvaluation<dim, degree, alt_q_points, 1, Number> phi(data,0);
    FEEvaluation<dim, degree, alt_q_points, 1, Number> alpha(data,0);
    FEEvaluation<dim, degree, alt_q_points, 1, Number> a(data,0);
    FEEvaluation<dim, degree, alt_q_points, 1, Number> pi(data,0);
    FEEvaluation<dim, degree, alt_q_points, 1, Number> psi(data,0);


    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        alpha.reinit(cell);
        alpha.gather_evaluate(alpha_solution, EvaluationFlags::values | EvaluationFlags::gradients);
        a.reinit(cell);
        a.gather_evaluate(a_solution, EvaluationFlags::values | EvaluationFlags::gradients);
        pi.reinit(cell);
        pi.gather_evaluate(pi_vec, EvaluationFlags::values | EvaluationFlags::gradients);
        psi.reinit(cell);
        psi.gather_evaluate(psi_solution, EvaluationFlags::values | EvaluationFlags::gradients);
        for (unsigned int q = 0; q < pi.n_q_points; ++q)
          {VectorizedArray<double> r_val = pi.quadrature_point(q)[0];

            Tensor<1, dim, VectorizedArray<double>> weak_form;
            weak_form[0] = -alpha.get_value(q)*psi.get_value(q)*pow_f.eval(a.get_value(q),-1);

            pi.submit_value(
                             2.*alpha.get_value(q)*psi.get_value(q)*pow_f.eval(a.get_value(q)*r_val,-1)
                            ,q);
            pi.submit_gradient(weak_form,q);

          }
        pi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,dst);
      }
  }


  template <int dim, int degree, int n_points_1d>
  void Pi_Operator<dim, degree, n_points_1d>::local_apply_face(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &v_vec,
    const std::pair<unsigned int, unsigned int> &     face_range) const
  {
    //This performs the integrations over faces of the rhs of pi evolution equation
    FEFaceEvaluation<dim, degree, alt_q_points> pi_in(data, true, 0);
    FEFaceEvaluation<dim, degree, alt_q_points> pi_out(data, false, 0);
    FEFaceEvaluation<dim, degree, alt_q_points> alpha_in(data, true, 0);
    FEFaceEvaluation<dim, degree, alt_q_points> alpha_out(data, false, 0);
    FEFaceEvaluation<dim, degree, alt_q_points> a_in(data, true, 0);
    FEFaceEvaluation<dim, degree, alt_q_points> a_out(data, false, 0);
    FEFaceEvaluation<dim, degree, alt_q_points> psi_in(data, true, 0);
    FEFaceEvaluation<dim, degree, alt_q_points> psi_out(data, false, 0);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        pi_in.reinit(face);
        pi_in.gather_evaluate(v_vec, EvaluationFlags::values);
        pi_out.reinit(face);
        pi_out.gather_evaluate(v_vec, EvaluationFlags::values);
        alpha_in.reinit(face);
        alpha_in.gather_evaluate(alpha_solution, EvaluationFlags::values);
        alpha_out.reinit(face);
        alpha_out.gather_evaluate(alpha_solution, EvaluationFlags::values);
        a_in.reinit(face);
        a_in.gather_evaluate(a_solution, EvaluationFlags::values);
        a_out.reinit(face);
        a_out.gather_evaluate(a_solution, EvaluationFlags::values);
        psi_in.reinit(face);
        psi_in.gather_evaluate(psi_solution, EvaluationFlags::values);
        psi_out.reinit(face);
        psi_out.gather_evaluate(psi_solution, EvaluationFlags::values);


        for (unsigned int q = 0; q < pi_in.n_q_points; ++q)
          {
            const VectorizedArray<Number> _avg =
                  (psi_in.get_value(q)*a_in.get_value(q)*pow_f.eval(alpha_in.get_value(q),-1)
                  +psi_out.get_value(q)*a_out.get_value(q)*pow_f.eval(alpha_out.get_value(q),-1));
            const VectorizedArray<Number> _jump =
                  (pi_in.get_value(q)
                  -pi_out.get_value(q));

            const VectorizedArray<Number> spd = 2.*alpha_out.get_value(q)*alpha_in.get_value(q)
                                                  *pow_f.eval(a_out.get_value(q)*alpha_in.get_value(q) + a_in.get_value(q)*alpha_out.get_value(q),-1);


            pi_in.submit_value(0.5*spd*alpha_in.get_value(q)*pow_f.eval(a_in.get_value(q),-1)*
            (_avg*pi_in.get_normal_vector(q)[0]
          - abs(pi_in.get_normal_vector(q)[0])*_jump)
                              ,q);
            pi_out.submit_value(-0.5*spd*alpha_out.get_value(q)*pow_f.eval(a_out.get_value(q),-1)*
            (_avg*pi_in.get_normal_vector(q)[0]
          - abs(pi_in.get_normal_vector(q)[0])*_jump)
                              ,q);
          }
        pi_in.integrate_scatter(EvaluationFlags::values, dst);
        pi_out.integrate_scatter(EvaluationFlags::values,dst);

    }
  }




  template <int dim, int degree, int n_points_1d>
  void Pi_Operator<dim, degree, n_points_1d>::local_apply_boundary_face(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &v_vec,
    const std::pair<unsigned int, unsigned int> &     face_range) const
  {
    //This performs the integrations over boundaries of the rhs of pi evolution equation
    FEFaceEvaluation<dim, degree, alt_q_points, 1, Number> pi(data, true, 0);
    FEFaceEvaluation<dim, degree, alt_q_points, 1, Number> psi(data, true, 0);
    FEFaceEvaluation<dim, degree, alt_q_points, 1, Number> alpha(data, true, 0);
    FEFaceEvaluation<dim, degree, alt_q_points, 1, Number> a(data, true, 0);
    FEFaceEvaluation<dim, degree, alt_q_points, 1, Number> phi(data, true, 0);
    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        pi.reinit(face);
        pi.gather_evaluate(v_vec, EvaluationFlags::values);
        psi.reinit(face);
        psi.gather_evaluate(psi_solution, EvaluationFlags::values);
        a.reinit(face);
        a.gather_evaluate(a_solution, EvaluationFlags::values);
        alpha.reinit(face);
        alpha.gather_evaluate(alpha_solution, EvaluationFlags::values);


        const auto boundary_id = data.get_boundary_id(face);

        for (unsigned int q = 0; q < pi.n_q_points; ++q)
          {
            const VectorizedArray<Number> spd = 2.*alpha.get_value(q)*alpha.get_value(q)
                                                  *pow_f.eval(a.get_value(q)*alpha.get_value(q) + a.get_value(q)*alpha.get_value(q),-1);
            if (boundary_id == 0){ // Origin
              pi.submit_value(0.,q);
            }
            else if (boundary_id == 1){ // Outer edge
              pi.submit_value(0.5*spd*alpha.get_value(q)*pow_f.eval(a.get_value(q),-1)*
            (psi.get_value(q)*a.get_value(q)*pow_f.eval(alpha.get_value(q),-1))
                              ,q);
            }
            else{
              AssertThrow(false, ExcMessage("Unknown boundary was used that did"
              "not match inbound or outbound boundary."))
            }
          }

        pi.integrate_scatter(EvaluationFlags::values, dst);
      }
  }


  template <int dim, int degree, int n_points_1d>
  void Pi_Operator<dim, degree, n_points_1d>::local_apply_inverse_mass_matrix(
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
  void Pi_Operator<dim, degree, n_points_1d>::apply(
    const double                                      current_time,
    const LinearAlgebra::distributed::Vector<Number> &src,
    LinearAlgebra::distributed::Vector<Number> &      dst) const
  {
    //Applies the operator to an input vector (src) and outputs result to dst
    {

      data.cell_loop(&Pi_Operator::local_apply_cell,
                &Pi_Operator::local_apply_face,
                &Pi_Operator::local_apply_boundary_face,
                this,
                dst,
                src,
                true);
    }

    {

      data.cell_loop(&Pi_Operator::local_apply_inverse_mass_matrix,
                     this,
                     dst,
                     dst);
    }
  }


  template <int dim, int degree, int n_points_1d>
  void Pi_Operator<dim, degree, n_points_1d>::perform_stage(
    const Number                                      /*current_time*/,
    const Number                                      factor_solution,
    const Number                                      factor_ai,
    const LinearAlgebra::distributed::Vector<Number> &current_ri,
    LinearAlgebra::distributed::Vector<Number> &      vec_ki,
    LinearAlgebra::distributed::Vector<Number> &      solution,
    LinearAlgebra::distributed::Vector<Number> &      next_ri) const
  {
    //Performs the stage of the runge kutta scheme
    //First applies the operator to a vector.
    //Then applies the inverse mass matrix to the result of the first part
    //Then updates the solution vector
    {

      data.loop(&Pi_Operator::local_apply_cell,
                &Pi_Operator::local_apply_face,
                &Pi_Operator::local_apply_boundary_face,
                this,
                vec_ki,
                current_ri,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }

    {
      data.cell_loop(
        &Pi_Operator::local_apply_inverse_mass_matrix,
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
