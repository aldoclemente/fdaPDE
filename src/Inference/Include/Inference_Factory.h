#ifndef __INFERENCE_FACTORY_H__
#define __INFERENCE_FACTORY_H__

// HEADERS
#include "Inference_Base.h"
#include "Wald.h"
#include "Speckman.h"
#include "Eigen_Sign_Flip.h"
#include "../../Global_Utilities/Include/Make_Shared.h"
#include <memory>
#include <map>

//! A Factory class: A class for the choice of implementation for the computation of inferential objects.
/* \tparam InputHandler RegressionData of the problem
 */
template<typename InputHandler>
class Inference_Factory
{
private:
  static std::map<std::string,std::shared_ptr<Inference_Base<InputHandler>>>& get_Factory_Store(void)
  {
    static std::map<std::string,std::shared_ptr<Inference_Base<InputHandler>>> factory_Store; // initialize the static map
    return factory_Store;
  }

public:
  //! A method that takes as parameter a string and builds a pointer to the right implementation object
  /*!
    \param implementation_type type of implementation required
    \param inverter_ class demanded to the computation of MatrixNoCov inverse
    \param inf_car_ inference carrier object
    \return std::unique_ptr to the chosen solver
  */
  static std::shared_ptr<Inference_Base<InputHandler>> create_inference_method(const std::string & implementation_type_, std::shared_ptr<Inverse_Base> inverter_, const Inference_Carrier<InputHandler> & inf_car_, UInt pos_impl_)
  {
    std::map<std::string,std::shared_ptr<Inference_Base<InputHandler>>> factory_Store=get_Factory_Store(); // Get the static factory
    
    if(implementation_type_=="wald"){
      auto It = factory_Store.find("wald");
      if(It==factory_Store.end()){
	factory_Store.insert(std::make_pair<std::string, std::shared_ptr<Inference_Base<InputHandler>>>("wald", fdaPDE::make_shared<Wald<InputHandler>>(inverter_, inf_car_, pos_impl_)));
      }else{
	It->second->setpos_impl(pos_impl_);
      }
      return factory_Store["wald"];
    }
    if(implementation_type_=="speckman"){
      auto It = factory_Store.find("speckman");
      if(It==factory_Store.end()){
	factory_Store.insert(std::make_pair<std::string, std::shared_ptr<Inference_Base<InputHandler>>>("speckman", fdaPDE::make_shared<Speckman<InputHandler>>(inverter_, inf_car_, pos_impl_)));
      }else{
	It->second->setpos_impl(pos_impl_);
      }
      return factory_Store["speckman"];
    }
    if(implementation_type_=="eigen-sign-flip"){
      auto It = factory_Store.find("eigen-sign-flip");
      if(It==factory_Store.end()){
	factory_Store.insert(std::make_pair<std::string, std::shared_ptr<Inference_Base<InputHandler>>>("eigen-sign-flip", fdaPDE::make_shared<Eigen_Sign_Flip<InputHandler>>(inverter_, inf_car_, pos_impl_)));
      }else{
	It->second->setpos_impl(pos_impl_);
      }
      return factory_Store["eigen-sign-flip"];
    }
    else // deafult Wald
      {
	Rprintf("Implementation not found, using wald");
	auto It = factory_Store.find("wald");
	if(It==factory_Store.end()){
	  factory_Store.insert(std::make_pair<std::string, std::shared_ptr<Inference_Base<InputHandler>>>("wald", fdaPDE::make_shared<Wald<InputHandler>>(inverter_, inf_car_, pos_impl_)));
	}else{
	  It->second->setpos_impl(pos_impl_);
	}
	return factory_Store["wald"];
      }
  }
};

#endif
