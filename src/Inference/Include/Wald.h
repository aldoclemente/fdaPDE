#ifndef __WALD_H__
#define __WALD_H__

// HEADERS
#include "../../FdaPDE.h"
#include "../../Regression/Include/Mixed_FE_Regression.h"
#include "../../Regression/Include/Regression_Data.h"
#include "../../Lambda_Optimization/Include/Optimization_Data.h"
#include "../../Lambda_Optimization/Include/Solution_Builders.h"
#include "Inference_Data.h"
#include "Inference_Carrier.h"
#include "Inverter.h"
#include "Inference_Base.h"
#include <memory>

// *** Wald Class ***
//! Hypothesis testing and confidence intervals using Wald implementation
/*!
  This class performes hypothesis testing and/or computes confidence intervals using a Wald-type approach. It contains a reference to an inverter, that manages to compute the invertion of matrixNoCov in an exact or non-exact way; It contains a reference to an Inference_Carrier object that wraps all the information needed to make inference. There is only one public method that calls the proper private methods to compute what is requested by the user.
*/
template<typename InputHandler, MatrixType>
class Wald_Base:public Inference_Base<InputHandler, MatrixType>{
private:
  MatrixXr S;						//!< Smoothing matrix 
  Real tr_S=0; 						//!< Trace of smoothing matrix, needed for the variance-covariance matrix (V) and eventually GCV computation
  Real sigma_hat_sq; 					//!< Estimator for the variance of the residuals (SSres/(n_obs-(q+tr_S)))
  bool is_S_computed = false;				//!< Boolean that tells whether S has been computed or not
  MatrixXr V;						//!< Variance-Covariance matrix of the beta parameters
  bool is_V_computed = false;				//!< Boolean that tells whether V has been computed or not
  virtual void compute_S(void) = 0;			//!< Method used to compute S, either in an exact or non-exact way 
  void compute_V(void);					//!< Method used to compute V
  VectorXr compute_pvalue(void) override;		//!< Method used to compute the pvalues of the tests 
  MatrixXv compute_CI(void) override;			//!< Method to compute the confidence intervals
  void compute_sigma_hat_sq(void);                      //!< Method to compute the estimator of the variance of the residuals 
  
public:
  // CONSTUCTOR
  Wald_Base()=delete;	//The default constructor is deleted
  Wald_Base(std::shared_ptr<Inverse_Base<MatrixType>> inverter_, const Inference_Carrier<InputHandler> & inf_car_, UInt pos_impl_):Inference_Base<InputHandler, MatrixType>(inverter_, inf_car_, pos_impl_){}; 
  
  virtual ~ Wald_Base(){};
  
  Real compute_GCV_from_inference(void) const override; //!< Needed to compute exact GCV in case Wald test is required and GCV exact is not provided by lambda optimization (Run after S computation)
  
  
  // GETTERS
  inline const MatrixXr * getSp (void) const {return &this->S;}      //!< Getter of Sp \return Sp
  inline const MatrixXr * getVp (void) const {return &this->V;}      //!< Getter of Vp \ return Vp
  
  void print_for_debug(void) const;
};

template<typename InputHandler, MatrixType>
class Wald_Exact:public Wald_Base<InputHandler, MatrixType>{
private: 
  void compute_S(void) override;
public:
  // CONSTUCTOR
  Wald_Exact()=delete;	//The default constructor is deleted
  Wald_Exact(std::shared_ptr<Inverse_Base<MatrixType>> inverter_, const Inference_Carrier<InputHandler> & inf_car_, UInt pos_impl_):Wald_Base<InputHandler, MatrixType>(inverter_, inf_car_, pos_impl_){}; 
};


template<typename InputHandler, MatrixType>
class Wald_Non_Exact:public Wald_Base<InputHandler, MatrixType>{
private: 
  void compute_S(void) override;
public:
  // CONSTUCTOR
  Wald_Non_Exact()=delete;	//The default constructor is deleted
  Wald_Non_Exact(std::shared_ptr<Inverse_Base<MatrixType>> inverter_, const Inference_Carrier<InputHandler> & inf_car_, UInt pos_impl_):Wald_Base<InputHandler, MatrixType>(inverter_, inf_car_, pos_impl_){}; 
};


#include "Wald_imp.h"

#endif
