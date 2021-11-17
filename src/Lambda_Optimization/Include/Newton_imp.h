#ifndef __NEWTON_IMP_H__
#define __NEWTON_IMP_H__


/*!
 \param x0 the initial guess for the optimization method
 \param tolerance the tolerance used as stopping criterion for the iterative optimization method
 \param max_iter the maximum number of iterations
 \param ch a reference to a Checker object, used to set the reason of termination of the iterations.
 \param GCV_v a reference to the vector of GCV values evaluated during the iterative procedure
 \param lambda_v a reference to the vector of lambda values explored during the iterative procedure
 \return std::pair<Tuple, UInt>, a pair which containns the optimal lambda found and the number of iterations to reach the tolerance
*/
template <typename Tuple, typename Hessian, typename ...Extensions>
std::pair<Tuple, UInt> Newton_ex<Tuple, Hessian, Extensions...>::compute (const Tuple & x0, const Real tolerance, const UInt max_iter, Checker & ch, std::vector<Real> & GCV_v, std::vector<Tuple> & lambda_v)
{
        Rprintf("Sto facendo Newton esatto\n");

       // Initialize the algorithm
       Tuple x_old;
       Tuple x      = x0;
       UInt  n_iter = 0;
       Real  error  = std::numeric_limits<Real>::infinity();

       // Debugging purpose
       // Rprintf("\n Starting Initializing lambda phase\n");

       Rprintf("\n Starting Newton's iterations");//: starting point lambda=%f\n",x);

       //only the first time applied here
       Real    fx  = this->F.evaluate_f(x);
       Tuple   fpx = this->F.evaluate_first_derivative (x);
       Hessian fsx = this->F.evaluate_second_derivative(x);

       while(n_iter < max_iter)
       {
               // Debugging purpose f(x)
               GCV_v.push_back(fx);
               lambda_v.push_back(x);

               if(Auxiliary<Tuple>::isNull(fsx))
               {
                       // Debug message
                       // std::cout << "Division by zero" << std::endl;
                       return {x, n_iter};
               }

               ++n_iter;


               x_old = x;
               Auxiliary<Tuple>::divide(fsx, fpx, x);
               x = x_old - x;

               if (!Auxiliary<Tuple>::isPositive(x))
               {
                       Rprintf("\nProbably monotone increasing GCV function\n");

                       fx = this->F.evaluate_f(x);
                       return {x_old, n_iter};
               }

               // Put here the updates in order to compute error on the correct derivative and to have z_hat updated for the solution

               fpx = this->F.evaluate_first_derivative (x);

               error = Auxiliary<Tuple>::residual(fpx);

               Rprintf("\nStep number %d  of EXACT-NEWTON: residual = %f\n", n_iter, error);

               if(error<tolerance)
               {
                       /* fpx=this->F.evaluate_f(x-0.5);
                       fsx=this->F.evaluate_f(x+0.5);
                       if (std::abs(fpx-fsx)/fsx<=0.01)
                               Rprintf("GCV has a non standard shape");*/

                       ch.set_tolerance();
                       fx = this->F.evaluate_f(x);
                       return {x, n_iter};
               }

               fx  = this->F.evaluate_f(x);
               fsx = this->F.evaluate_second_derivative(x);
       }

       fx = this->F.evaluate_f(x);
       ch.set_max_iter();
       return {x, n_iter};
}


/*!
 \param x0 the initial guess for the optimization method
 \param tolerance the tolerance used as stopping criterion for the iterative optimization method
 \param max_iter the maximum number of iterations
 \param ch a reference to a Checker object, used to set the reason of termination of the iterations.
 \param GCV_v a reference to the vector of GCV values evaluated during the iterative procedure
 \param lambda_v a reference to the vector of lambda values explored during the iterative procedure
 \return std::pair<Real, UInt>, a pair which containns the optimal lambda found and the number of iterations to reach the tolerance
*/
template <typename ...Extensions>
std::pair<lambda::type<1>, UInt> Newton_fd<lambda::type<1>, Real, Extensions...>::compute (const lambda::type<1> & x0, const Real tolerance, const UInt max_iter, Checker & ch, std::vector<Real> & GCV_v, std::vector<lambda::type<1>> & lambda_v)
{
        Rprintf("Sto facendo Newton_fd nel caso 1D\n");

        // Initialize the algorithm
        Real x_old;
        Real x      = x0;
        UInt n_iter = 0;
        Real error  = std::numeric_limits<Real>::infinity();
        Real h      = 4e-8; //passo modificato

        
        Rprintf("\n Starting Newton's iterations: starting point lambda=%e\n",x);

        // Only the first time applied here
        // Rprintf("Forward: \n");
        Real fxph = this->F.evaluate_f(x+h);
        //Rprintf("Backward: \n");
        Real fxmh = this->F.evaluate_f(x-h);
        // Rprintf("Center: \n");
        Real fx  = this->F.evaluate_f(x);

        Real fpx = (fxph-fxmh)/(2*h);
        // Rprintf("fp(x): %f\n", fpx);

        Real fsx = (fxph+fxmh-(2*fx))/(h*h);
        // Rprintf("fs(x): %f\n", fsx);
        
        Rprintf("x=%f,fx:%f,fpx:%f,fsx:%f\n\tex_fpx:%f,ex_fsx:%f",x,
        	fx, fpx, fsx, this->F.evaluate_first_derivative(x), this->F.evaluate_second_derivative(x));

        while(n_iter < max_iter)
        {
                GCV_v.push_back(fx);
                lambda_v.push_back(x);

                //Debugging purpose f(x)
                if (Auxiliary<Real>::isNull(fsx))
                {
                        // Debug message
                        // std::cout << "Division by zero" << std::endl;
                        return {x, n_iter};
                }

                ++n_iter;


                x_old = x;
                Auxiliary<Real>::divide(fsx, fpx, x);
                x = x_old - x;

                if (x<=0)
                { //too small value
                        Rprintf("\nProbably monotone increasing GCV function\n");

                        fx = this->F.evaluate_f(x_old);
                        return {x_old, n_iter};
                }

                // Put here the updates in order to compute error on the correct derivative and to have z_hat updated for the solution
                // Rprintf("Forward:\n");
                fxph = this->F.evaluate_f(x+h);
                // Rprintf("Backward: \n");
                fxmh = this->F.evaluate_f(x-h);

                fpx = (fxph-fxmh)/(2*h);

                error = Auxiliary<Real>::residual(fpx);

                Rprintf("\nStep number %d  of FD-NEWTON: residual = %f\n", n_iter, error);

                if (error < tolerance)
                {
                        ch.set_tolerance();
                        fx  = this->F.evaluate_f(x); //eventuale miglioramento: va fatto altirmenti prende gli z:hat di quellos sbagliato.
                        return {x, n_iter};
                }

                // Rprintf("Center: \n");
                fx  = this->F.evaluate_f(x);

                fsx = (fxph+fxmh-(2*fx))/(h*h);
                
        	Rprintf("x=%f,fx:%f,fpx:%f,fsx:%f\n\tex_fpx:%f,ex_fsx:%f",x,
        	fx, fpx, fsx, this->F.evaluate_first_derivative(x), this->F.evaluate_second_derivative(x));

                // Rprintf("fp(x): %f\n", fpx);
                // Rprintf("fs(x): %f\n", fsx);
        }
        fx  = this->F.evaluate_f(x);
        ch.set_max_iter();
        return {x, n_iter};
}

template <typename ...Extensions>
std::pair<lambda::type<2>, UInt> Newton_fd<lambda::type<2>, MatrixXr, Extensions...>::compute (const lambda::type<2> & x0, const Real tolerance, const UInt max_iter, Checker & ch, std::vector<Real> & GCV_v, std::vector<lambda::type<2>> & lambda_v)
{
        Rprintf("\nStiamo runnando newton_fd nel caso 2d\n");
        // Initialize the algorithm
        lambda::type<2> x_old;
        lambda::type<2> x = x0;
        UInt n_iter = 0;
        Real error  = std::numeric_limits<Real>::infinity();
        Real h      = 4e-8;
        
        Rprintf("\n Starting Newton's iterations: starting point lambda=(%e,%e)\n",x(0),x(1));


        // Only the first time applied here
        // Rprintf("Forward: \n");
        Rprintf("A");
        Real space_fxph = this->F.evaluate_f(lambda::make_pair(x(0)+h, x(1)));
        Rprintf("B");
        Real time_fxph = this->F.evaluate_f(lambda::make_pair(x(0), x(1)+h));
        Rprintf("C");
        //Rprintf("Backward: \n");
        Real space_fxmh = this->F.evaluate_f(lambda::make_pair(x(0)-h, x(1)));
        Rprintf("D");
        Real time_fxmh = this->F.evaluate_f(lambda::make_pair(x(0), x(1)-h));
        Rprintf("E");
        // Rprintf("Center: \n");
        Real fx  = this->F.evaluate_f(x); //space_fx = time_fx (no h movements)
        Rprintf("F");

        Real space_fpx = (space_fxph-space_fxmh)/(2*h);
        Real time_fpx = (time_fxph-time_fxmh)/(2*h);
       	lambda::type<2> fpx = lambda::make_pair(space_fpx, time_fpx);
        // Rprintf("fp(x): %f\n", fpx);

        Real space_fsx = (space_fxph+space_fxmh-(2*fx))/(h*h);
        Real time_fsx = (time_fxph+time_fxmh-(2*fx))/(h*h);
        Rprintf("G");
        //https://scicomp.stackexchange.com/questions/11294/2nd-order-centered-finite-difference-approximation-of-u-xy
        Rprintf("Inizio a calcolare la derivata mista fd\n");
        Real mixed_fsx = (this->F.evaluate_f(lambda::make_pair(x(0)+h, x(1)+h))-
        		this->F.evaluate_f(lambda::make_pair(x(0)+h, x(1)-h))-
        		this->F.evaluate_f(lambda::make_pair(x(0)-h, x(1)+h))+
        		this->F.evaluate_f(lambda::make_pair(x(0)-h, x(1)-h)))/(4*h*h);
	Rprintf("H");
        MatrixXr fsx(2,2);
        fsx(0,0) = space_fsx; fsx(0,1) = mixed_fsx;
        fsx(1,0) = mixed_fsx; fsx(1,1) = time_fsx;
        // Rprintf("fs(x): %f\n", fsx);
        
        
        Rprintf("x=(%f,%f)\nfx=%f\ndfx=(%f,%f)\nddfx(%f,%f,%f)\n",
        	x(0), x(1), fx, space_fpx, time_fpx, space_fsx, time_fsx, mixed_fsx);
	Rprintf("I");
        lambda::type<2> prima = this->F.evaluate_first_derivative(x);
        Rprintf("J");
        Rprintf("exact\tdfx=(%f,%f)\n\n\n", prima(0), prima(1));
        MatrixXr seconda = this->F.evaluate_second_derivative(x);
	Rprintf("exact\tddfx(%f,%f,%f)\n\n\n", seconda.coeff(0,0), seconda.coeff(1,1), seconda.coeff(0,1));
	Rprintf("K");
        	
        while(n_iter < max_iter)
        {
                Rprintf("\nNewton iteration %d \n", n_iter);
                GCV_v.push_back(fx);
                lambda_v.push_back(x);

                //Debugging purpose f(x)
                if (Auxiliary<lambda::type<2>>::isNull(fsx))
                {
                        // Debug message
                        // std::cout << "Division by zero" << std::endl;
                        return {x, n_iter};
                }

                ++n_iter;


                x_old = x;
                Auxiliary<lambda::type<2>>::divide(fsx, fpx, x);
                x = x_old - x;

                if (x(0)<=0||x(1)<=0)
                { //too small value
                        Rprintf("\nProbably monotone increasing GCV function\n");

                        fx = this->F.evaluate_f(x_old);
                        return {x_old, n_iter};
                }

                // Put here the updates in order to compute error on the correct derivative and to have z_hat updated for the solution
                // Rprintf("Forward:\n");
		Real space_fxph = this->F.evaluate_f(lambda::make_pair(x(0)+h, x(1)));
		Real time_fxph = this->F.evaluate_f(lambda::make_pair(x(0), x(1)+h));
		//Rprintf("Backward: \n");
		Real space_fxmh = this->F.evaluate_f(lambda::make_pair(x(0)-h, x(1)));
		Real time_fxmh = this->F.evaluate_f(lambda::make_pair(x(0), x(1)-h));
		// Rprintf("Center: \n");
		Real fx  = this->F.evaluate_f(x); //space_fx = time_fx (no h movements)

		Real space_fpx = (space_fxph-space_fxmh)/(2*h);
		Real time_fpx = (time_fxph-time_fxmh)/(2*h);
	       	lambda::type<2> fpx = lambda::make_pair(space_fpx, time_fpx);
		// Rprintf("fp(x): %f\n", fpx);

                error = Auxiliary<lambda::type<2>>::residual(fpx);

                Rprintf("\nStep number %d  of FD-NEWTON: residual = %f\n", n_iter, error);

                if (error < tolerance)
                {
                        ch.set_tolerance();
                        fx  = this->F.evaluate_f(x); //eventuale miglioramento: va fatto altirmenti prende gli z:hat di quellos sbagliato.
                        return {x, n_iter};
                }

                // Rprintf("Center: \n");
                fx  = this->F.evaluate_f(x);

                Real space_fsx = (space_fxph+space_fxmh-(2*fx))/(h*h);
		Real time_fsx = (time_fxph+time_fxmh-(2*fx))/(h*h);
		//https://scicomp.stackexchange.com/questions/11294/2nd-order-centered-finite-difference-approximation-of-u-xy
		Real mixed_fsx = (this->F.evaluate_f(lambda::make_pair(x(0)+h, x(1)+h))-
				this->F.evaluate_f(lambda::make_pair(x(0)+h, x(1)-h))-
				this->F.evaluate_f(lambda::make_pair(x(0)-h, x(1)+h))+
				this->F.evaluate_f(lambda::make_pair(x(0)-h, x(1)-h)))/(4*h*h);
		MatrixXr fsx(2,2);
		fsx(0,0) = space_fsx; fsx(0,1) = mixed_fsx;
		fsx(1,0) = mixed_fsx; fsx(1,1) = time_fsx;
		//Rprintf("fs(x): %f\n", fsx);

                //Rprintf("fp(x): %f\n", fpx);
                //Rprintf("fs(x): %f\n", fsx);
                
                Rprintf("x=(%f,%f)\nfx=%f\ndfx=(%f,%f)\nddfx(%f,%f,%f)\n",
        		x(0), x(1), fx, space_fpx, time_fpx, space_fsx, time_fsx, mixed_fsx);
		lambda::type<2> prima = this->F.evaluate_first_derivative(x);
		Rprintf("exact\tdfx=(%f,%f)\n\n\n",
			prima(0), prima(1));
		MatrixXr seconda = this->F.evaluate_second_derivative(x);
		Rprintf("exact\tdfx=(%f,%f)\nddfx(%f,%f,%f)\n\n\n",
        	prima(0), prima(1), seconda.coeff(0,0), seconda.coeff(1,1), seconda.coeff(0,1));
        }
        fx  = this->F.evaluate_f(x);
        ch.set_max_iter();
        return {x, n_iter};
}

#endif
