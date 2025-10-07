// This file just provides some useful functions that are used throughout the files.
// Many are needed just to account for the VectorizedArray type in dealii

namespace Scalar_Evolution
{
  using namespace dealii;
  
  template <int dim>
  class phi_init:public Function<dim>
  //This just is the intial condition for the phi field.
  {
  public:
    virtual double value(const Point<dim> &p_vectorized,
                         const unsigned int component = 0) const override;

    template <typename number>
    VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p_vectorized,
                 const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> phi_init<dim>::value(const Point<dim, VectorizedArray<number>> &p_vectorized,
                                 const unsigned int /*component*/) const
  {
    double amp =seed_amp;
    double sig = seed_sigma;
    double r_b = seed_rb;
    VectorizedArray<number> result;
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
          p[d] = p_vectorized[d][v];
        double rho = p[0];
        result[v] = amp*exp(-(rho-r_b)*(rho-r_b)/(sig*sig));
      }

    return result;
  }

  template <int dim>
  double phi_init<dim>::value(const Point<dim> & p,
                                 const unsigned int /*component*/) const
  {
    double amp = seed_amp;
    double sig = seed_sigma;
    double r_b = seed_rb;
    double rho = p[0];
    return amp*exp(-(rho-r_b)*(rho-r_b)/(sig*sig));
  }






  template <int dim>
  class psi_init:public Function<dim>
  //Again jus the initial condition but for psi this time. (the spatial derivative of the scalar field)
  {
  public:
    virtual double value(const Point<dim> &p_vectorized,
                         const unsigned int component = 0) const override;

    template <typename number>
    VectorizedArray<number> value(const Point<dim, VectorizedArray<number>> &p_vectorized,
                 const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> psi_init<dim>::value(const Point<dim, VectorizedArray<number>> &p_vectorized,
                                 const unsigned int /*component*/) const
  {
    double amp =seed_amp;
    double sig = seed_sigma;
    double r_b = seed_rb;
    VectorizedArray<number> result;
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
          p[d] = p_vectorized[d][v];
        double rho = p[0];
        result[v] = -2.*(rho-r_b)/(sig*sig)*amp*exp(-(rho-r_b)*(rho-r_b)/(sig*sig));
      }

    return result;
  }

  template <int dim>
  double psi_init<dim>::value(const Point<dim> & p,
                                 const unsigned int /*component*/) const
  {
    double amp = seed_amp;
    double sig = seed_sigma;
    double r_b = seed_rb;
    double rho = p[0];
    return -2.*(rho-r_b)/(sig*sig)*amp*exp(-(rho-r_b)*(rho-r_b)/(sig*sig));
  }




  // Allows for multiplications of tensors/vectors
  template <int n_components, int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
    Tensor<1, n_components, Number>
    operator*(const Tensor<1, n_components, Tensor<1, dim, Number>> &matrix,
              const Tensor<1, dim, Number> &                         vector)
  {
    Tensor<1, n_components, Number> result;
    for (unsigned int d = 0; d < n_components; ++d)
      result[d] = matrix[d] * vector;
    return result;
  }


  // This is for dividing when a Vectorized array is used. For consistency it
  // also accounts for doubles. It can be used for any exponentiation. Dividing
  // just needs an negative power for the exponent.
  template <int dim>
  class pow_func:public Function<dim>
  {
  public:
    double eval(const double &field,
                         const int &power,
                         const unsigned int component = 0) const;

    template <typename number>
    VectorizedArray<number> eval(const VectorizedArray<number> &field,
                 const int &power,
                 const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> pow_func<dim>::eval(const VectorizedArray<number> &field,
                                 const int &power,
                                 const unsigned int /*component*/) const
  {
    VectorizedArray<number> result;
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        result[v] = std::pow(field[v],power);
      }

    return result;
  }

  template <int dim>
  double pow_func<dim>::eval(const double &field,
                              const int &power,
                                 const unsigned int /*component*/) const
  {
    return std::pow(field,power);
  }



  // This is for using the exponential function and accounts for VectorizedArray
  template <int dim>
  class exp_func:public Function<dim>
  {
  public:
    virtual double eval(const double &field,
                         const unsigned int component = 0) const;

    template <typename number>
    VectorizedArray<number> eval(const VectorizedArray<number> &field,
                 const unsigned int       component = 0) const;
  };

  template <int dim>
  template <typename number>
  VectorizedArray<number> exp_func<dim>::eval(const VectorizedArray<number> &field,
                                 const unsigned int /*component*/) const
  {
    VectorizedArray<number> result;
    for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
      {
        result[v] = std::exp(field[v]);
      }

    return result;
  }

  template <int dim>
  double exp_func<dim>::eval(const double &field,
                                 const unsigned int /*component*/) const
  {
    return std::exp(field);
  }


  // These are self explanatory functions only used in the filtering of results.
  double factorial(const unsigned int &x){
    double result = 1;
    for (unsigned int i = x; i > 0; --i){
      result = result * i;
    }
    return result;
  }

  template <typename T>
  int sign_of(T val){
    return (T(0) < val) - (val < T(0));
  }

}//namespace Brill_Evolution
