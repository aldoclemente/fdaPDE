#ifndef __AUXILIARY_SKELETON_H__
#define __AUXILIARY_SKELETON_H__

#include "../../FdaPDE.h"
#include "../../FE_Assemblers_Solvers/Include/Finite_Element.h"
#include "../../FE_Assemblers_Solvers/Include/Matrix_Assembler.h"
#include "../../Mesh/Include/Mesh.h"

template<UInt ORDER, UInt mydim, UInt ndim>
SEXP get_integration_points_skeleton(SEXP Rmesh)
{
	using Integrator = typename FiniteElement<ORDER, mydim, ndim>::Integrator;
	using meshElement = typename MeshHandler<ORDER, mydim, ndim>::meshElement;

	MeshHandler<ORDER, mydim, ndim> mesh(Rmesh);

	SEXP result;
	PROTECT(result=Rf_allocVector(REALSXP, ndim * Integrator::NNODES * mesh.num_elements()));
	for(UInt i=0; i<mesh.num_elements(); ++i){
		meshElement el = mesh.getElement(i);
		for(UInt l = 0; l < Integrator::NNODES; ++l){
			Point<ndim> p{el.getM_J() * Integrator::NODES[l].eigenView()};
			p += el[0];
			for(UInt j=0; j < ndim; ++j)
				REAL(result)[j * mesh.num_elements() * Integrator::NNODES + i * Integrator::NNODES + l] = p[j];
		}
	}

	UNPROTECT(1);
	return(result);
}

template<UInt ORDER, UInt mydim, UInt ndim, typename A>
SEXP get_FEM_Matrix_skeleton(SEXP Rmesh, EOExpr<A> oper)
{
	MeshHandler<ORDER, mydim, ndim> mesh(Rmesh);

	FiniteElement<ORDER, mydim, ndim> fe;

	SpMat AMat;
	Assembler::operKernel(oper, mesh, fe, AMat);

	//Copy result in R memory
	SEXP result;
	result = PROTECT(Rf_allocVector(VECSXP, 2));
	SET_VECTOR_ELT(result, 0, Rf_allocMatrix(INTSXP, AMat.nonZeros() , 2));
	SET_VECTOR_ELT(result, 1, Rf_allocVector(REALSXP, AMat.nonZeros()));

	int *rans = INTEGER(VECTOR_ELT(result, 0));
	Real  *rans2 = REAL(VECTOR_ELT(result, 1));
	UInt i = 0;
	for (UInt k=0; k < AMat.outerSize(); ++k)
		{
			for (SpMat::InnerIterator it(AMat,k); it; ++it)
			{
				//std::cout << "(" << it.row() <<","<< it.col() <<","<< it.value() <<")\n";
				rans[i] = 1+it.row();
				rans[i + AMat.nonZeros()] = 1+it.col();
				rans2[i] = it.value();
				i++;
			}
		}
	UNPROTECT(1);
	return(result);
}

template<UInt ORDER, UInt mydim, UInt ndim>
SEXP get_psi_matrix_skeleton(SEXP Rmesh, SEXP Rlocations){
    
    using meshElement = typename MeshHandler<ORDER, mydim, ndim>::meshElement;
    MeshHandler<ORDER, mydim, ndim> mesh(Rmesh);
    RNumericMatrix locations(Rlocations);
        
    UInt nlocs = locations.nrows();
    constexpr UInt EL_NNODES = how_many_nodes(ORDER,mydim);
    UInt nnodes = mesh.num_nodes();	
    Element<EL_NNODES,mydim,ndim> current_element;
    Point<ndim> current_point;
    SpMat psi_ = SpMat(nlocs, nnodes);
    
    for (UInt i = 0; i < nlocs; ++i) {
        std::array<Real, ndim> coords;
        for(UInt n=0; n<ndim; ++n) coords[n] = locations(i,n);

        current_point = Point<ndim>(coords);
        current_element = mesh.findLocationNaive(current_point);
        
        if(current_element.getId() == Identifier::NVAL) continue;
        
        for (int j=0; j < EL_NNODES; ++j) {
            Real value = current_element.evaluate_point(current_point, Eigen::Matrix<Real,EL_NNODES,1>::Unit(j));
            psi_.insert(i, current_element[j].getId()) =  value;
         }
    }//end of for loop
    
    SEXP result;
	result = PROTECT(Rf_allocVector(VECSXP, 2));
	SET_VECTOR_ELT(result, 0, Rf_allocMatrix(INTSXP, psi_.nonZeros() , 2));
	SET_VECTOR_ELT(result, 1, Rf_allocVector(REALSXP, psi_.nonZeros()));

    int *rans = INTEGER(VECTOR_ELT(result, 0));
	Real  *rans2 = REAL(VECTOR_ELT(result, 1));
	UInt i = 0;
	for (UInt k=0; k < psi_.outerSize(); ++k){
	    for (SpMat::InnerIterator it(psi_,k); it; ++it){
				rans[i] = 1+it.row();
				rans[i + psi_.nonZeros()] = 1+it.col();
				rans2[i] = it.value();
				i++;
		}
	}
	UNPROTECT(1);
	return(result);
}

#endif
