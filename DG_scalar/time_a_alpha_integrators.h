namespace Scalar_Evolution
{
  using namespace dealii;  
  // These are just some low storage runge kutta schemes provided by deal.II
  // lsrk_scheme is the scheme that is used for the time integration.
  enum LowStorageRungeKuttaScheme
  {
    stage_3_order_3, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_5_order_4, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_7_order_4, /* Tselios, Simos, 2007 */
    stage_9_order_5, /* Kennedy, Carpenter, Lewis, 2000 */
    FE,
    RK_4
  };
  constexpr LowStorageRungeKuttaScheme lsrk_scheme = stage_5_order_4;



  // This just defines all the necessary information for all the lsrk schemes
  class LowStorageRungeKuttaIntegrator
  {
  public:
    LowStorageRungeKuttaIntegrator(const LowStorageRungeKuttaScheme scheme)
    {
      TimeStepping::runge_kutta_method lsrk;
      // First comes the three-stage scheme of order three by Kennedy et al.
      // (2000). While its stability region is significantly smaller than for
      // the other schemes, it only involves three stages, so it is very
      // competitive in terms of the work per stage.
      switch (scheme)
        {
          case stage_3_order_3:
            {
              lsrk = TimeStepping::LOW_STORAGE_RK_STAGE3_ORDER3;
              TimeStepping::LowStorageRungeKutta<
                LinearAlgebra::distributed::Vector<Number>>
                rk_integrator(lsrk);
              rk_integrator.get_coefficients(ai, bi, ci);
              break;
            }

            // The next scheme is a five-stage scheme of order four, again
            // defined in the paper by Kennedy et al. (2000).
          case stage_5_order_4:
            {
              lsrk = TimeStepping::LOW_STORAGE_RK_STAGE5_ORDER4;
              TimeStepping::LowStorageRungeKutta<
                LinearAlgebra::distributed::Vector<Number>>
                rk_integrator(lsrk);
              rk_integrator.get_coefficients(ai, bi, ci);
              break;
            }

            // The following scheme of seven stages and order four has been
            // explicitly derived for acoustics problems. It is a balance of
            // accuracy for imaginary eigenvalues among fourth order schemes,
            // combined with a large stability region. Since DG schemes are
            // dissipative among the highest frequencies, this does not
            // necessarily translate to the highest possible time step per
            // stage. In the context of the present tutorial program, the
            // numerical flux plays a crucial role in the dissipation and thus
            // also the maximal stable time step size. For the modified
            // Lax--Friedrichs flux, this scheme is similar to the
            // `stage_5_order_4` scheme in terms of step size per stage if only
            // stability is considered, but somewhat less efficient for the HLL
            // flux.
          case stage_7_order_4:
            {
              lsrk = TimeStepping::LOW_STORAGE_RK_STAGE7_ORDER4;
              TimeStepping::LowStorageRungeKutta<
                LinearAlgebra::distributed::Vector<Number>>
                rk_integrator(lsrk);
              rk_integrator.get_coefficients(ai, bi, ci);
              break;
            }

            // The last scheme included here is the nine-stage scheme of order
            // five from Kennedy et al. (2000). It is the most accurate among
            // the schemes used here, but the higher order of accuracy
            // sacrifices some stability, so the step length normalized per
            // stage is less than for the fourth order schemes.
          case stage_9_order_5:
            {
              lsrk = TimeStepping::LOW_STORAGE_RK_STAGE9_ORDER5;
              TimeStepping::LowStorageRungeKutta<
                LinearAlgebra::distributed::Vector<Number>>
                rk_integrator(lsrk);
              rk_integrator.get_coefficients(ai, bi, ci);
              break;
            }

            //Finally, this is a Forward Euler scheme made by me for
            // testing purposes and debugging. No real reason to use it.
          case FE:
          {
            lsrk = TimeStepping::FORWARD_EULER;
            TimeStepping::ExplicitRungeKutta<
              LinearAlgebra::distributed::Vector<Number>>
              rk_integrator(lsrk);
              ai.push_back(0);
              bi.push_back(1);
              //bi.push_back(0);
              ci.push_back(0);
            //rk_integrator.get_coefficients(ai, bi, ci);
            break;
          }

          default:
            AssertThrow(false, ExcNotImplemented());
        }

    }

    unsigned int n_stages() const
    {
      return bi.size();
    }

      // This is the integrator for the a and alpha solutions.
      // a and alpha are hard coded to do RK4 methods within the
      // perform stage function of their respective operators (
      // fake_a_operator and fake_alpha_operator). This just 
      // passes the variables that they need along. It isn't
      // the cleanest way but it works.

     template <typename Operator_0, int dim>
     void perform_integration_a(const Operator_0 &pde_operator_0,
                            const double    current_time,
                            const double    time_step,
                            double &solution_0,
                            double &vec_ri_0,
                            const FESystem<dim> &fe_DG,
                            const unsigned int &dof_num,
                            DoFHandler<dim> &dof_handler_DG) const
    {

      pde_operator_0.perform_stage(current_time,
                                 time_step,
                                 time_step,
                                 solution_0,
                                 solution_0,
                                 vec_ri_0,
                                 fe_DG,
                                 dof_num,
                                 dof_handler_DG);
     }

    // The main function of the time integrator is to go through the stages
    // of the time integration scheme. It goes through each of the operators:
    // phi, pi, psi, and other_a. (Other a was just evolving the a_solution
    // using the a_operator it isn't as stable it was just for debugging).
    // The first stage is done seperately as it has slightly modified inputs.
    // Then, before all future stages the a and alpha solutions are updated
    // to match the partial runge kutta step forward. All the phi_solution,
    // pi_solution, psi_solution, and other_a_solution vectors are used to hold
    // the intermediary solutions for each stage of the time integration.
    // Again perform stage is used for each operator that is defined in their
    // own files.
    template <typename VectorType, typename Operator_0,typename Operator_1,
              typename Operator_2,typename Operator_3,typename Operator_4,
              typename Operator_5, int dim>
    void perform_time_step(const Operator_0 &pde_operator_0,
                           const Operator_1 &pde_operator_1,
                           const Operator_2 &pde_operator_2,
                           const Operator_5 &pde_operator_3,
                           const Operator_3 &fake_a_operator,
                           const Operator_4 &fake_alpha_operator,
                           const double    current_time,
                           const double    time_step,
                           std::vector<VectorType> &    solution,
                           std::vector<VectorType> &    vec_ri,
                           VectorType &    vec_ki,
                           DoFHandler<dim> &dof_handler_DG,
                           const FESystem<dim> &fe_DG) const
    {

    const LowStorageRungeKuttaIntegrator integrator(lsrk_scheme);
    //Performing the first stage of the time integration

      pde_operator_0.perform_stage(current_time,
                                 bi[0] * time_step,
                                 ai[0] * time_step,
                                 solution[0],
                                 vec_ri[0],
                                 solution[0],
                                 vec_ri[0]);
       pde_operator_1.perform_stage(current_time,
                                  bi[0] * time_step,
                                  ai[0] * time_step,
                                  solution[1],
                                  vec_ri[1],
                                  solution[1],
                                  vec_ri[1]);
        pde_operator_2.perform_stage(current_time,
                                   bi[0] * time_step,
                                   ai[0] * time_step,
                                   solution[2],
                                   vec_ri[2],
                                   solution[2],
                                   vec_ri[2]);

         pde_operator_3.perform_stage(current_time,
                                    bi[0] * time_step,
                                    ai[0] * time_step,
                                    solution[3],
                                    vec_ri[3],
                                    solution[3],
                                    vec_ri[3]);
      //Saving the intermediary solutions
      phi_solution = vec_ri[0];
      pi_solution = vec_ri[1];
      psi_solution = vec_ri[2];
      other_a_solution = vec_ri[3];

      for (unsigned int stage = 1; stage < bi.size(); ++stage)
        {
          //Updating the a and alpha solutions to match the partial runge kutta step forward
          // a is integrated outward and then alpha is integrated inward
          double time_step_a;

          double fake_time = 0;

          a_solution[0] = 1.;
          double a_val = 1.;
          double rk_a = 1.;
          // a is integrated outward
          for (unsigned int i=1; i<dof_handler_DG.n_dofs(); ++i){
              time_step_a = dof_r_val[i][0] - fake_time;
              while(fake_time < dof_r_val[i][0]){
                integrator.perform_integration_a(fake_a_operator,
                                               fake_time,
                                               time_step_a,
                                               a_val,
                                               rk_a,
                                               fe_DG,
                                               i,
                                               dof_handler_DG);
               fake_time += time_step_a;
               rk_a = a_val;
              }
              a_solution.local_element(i) = a_val;
          }

           double alpha_val = 1./a_val;
           alpha_solution(dof_handler_DG.n_dofs() - 1) = alpha_val;
           rk_a = alpha_val;
          // alpha is integrated inward
           for (unsigned int i=dof_handler_DG.n_dofs()-1; i>0; --i){
               time_step_a = dof_r_val[i-1][0] - fake_time;
               while(fake_time > dof_r_val[i-1][0]){
                 integrator.perform_integration_a(fake_alpha_operator,
                                                fake_time,
                                                time_step_a,
                                                alpha_val,
                                                rk_a,
                                                fe_DG,
                                                i,
                                                dof_handler_DG);
                fake_time += time_step_a;
                rk_a = alpha_val;
               }
               alpha_solution[i-1] = alpha_val;
           }

          //Now all future stages are performed in the time integration
          const double c_i = ci[stage];
          pde_operator_0.perform_stage(current_time + c_i * time_step,
                                     bi[stage] * time_step,
                                     (stage == bi.size() - 1 ?
                                        0 :
                                        ai[stage] * time_step),
                                     vec_ri[0],
                                     vec_ki,
                                     solution[0],
                                     vec_ri[0]);

           pde_operator_1.perform_stage(current_time + c_i * time_step,
                                      bi[stage] * time_step,
                                      (stage == bi.size() - 1 ?
                                         0 :
                                         ai[stage] * time_step),
                                      vec_ri[1],
                                      vec_ki,
                                      solution[1],
                                      vec_ri[1]);

          pde_operator_2.perform_stage(current_time + c_i * time_step,
                                     bi[stage] * time_step,
                                     (stage == bi.size() - 1 ?
                                        0 :
                                        ai[stage] * time_step),
                                     vec_ri[2],
                                     vec_ki,
                                     solution[2],
                                     vec_ri[2]);

         pde_operator_3.perform_stage(current_time + c_i * time_step,
                                    bi[stage] * time_step,
                                    (stage == bi.size() - 1 ?
                                       0 :
                                       ai[stage] * time_step),
                                    vec_ri[3],
                                    vec_ki,
                                    solution[3],
                                    vec_ri[3]);


        // intermediary values are updated
         phi_solution = vec_ri[0];
         pi_solution = vec_ri[1];
         psi_solution = vec_ri[2];
         other_a_solution = vec_ri[3];
       }
       // finally the solution of a full runge kutta solution is saved to its respective
       // solution vector after all stages have finished
       phi_solution = solution[0];
       pi_solution = solution[1];
       psi_solution = solution[2];
       other_a_solution = solution[3];

}
  private:
    std::vector<double> bi;
    std::vector<double> ai;
    std::vector<double> ci;
  };
}