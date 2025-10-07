This is a code for solving numerically the collapse of massless scalar fields
in spherical symmetry using Discontinuous Galerkin methods in dealii.

Dealii would need to be compiled separately and is not included here.

This is done to replicate results from a seminal paper by Matthew Choptuik
found here: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.70.9

## Outline

- scalar_field_engine.cc is the main file that runs everything.
	it creates the mesh, initializes degrees of freedom, 
	initializes operators, and calls the needed functions
	to form the initial conditions and eventually to evolve
	the solutions. Has a bit list of global variables at the
	top of the file for slightly tweaking the conditions
	of the scalar field and how it is solved.
	
- fake_a_operator.h contains the functions for integrating the
	equation for solving for a outwards from the origin
	
- fake_alpha_operator.h contains the functions for solving the
	equation for alpha by inwards from the outer edge
	
- phi_operator.h contains functions for evolving phi solution
	(Should be able to remove this and mentions of it
	in scalar_field_engine.cc and everything would still
	work. Just included for visualization)

- psi_operator.h contains functions for evolving psi solution

- pi_operator.h contains functions for evolving pi solution

- time_a_alpha_integrators.h contains the loop for performing
	runge kutta integration of solutions and the variable
	for specifying the scheme to be used.
	Also, contains a funciton that passes information to
	the fake_a_operator and fake_alpha_operator but
	doesn't do much besides pass the information along.
	
- real_a_operator.h contains functions for evolving a solution
	This is more just for comparison as the values computed
	by fake_a_operator are more stable
	(Should be able to remove this and mentions of it
	in scalar_field_engine.cc and everything would still
	work. Just included for visualization)
	
- helper_functions.h contains some functions that are useful in 
	the code. Such as the actual functions that define the
	initial conditions of phi and psi. (Pi is just 0
	everywhere initially). Along with some functions for
	dividing unique data types from dealii.
	
-OE_operator.h contains functions for implimenting oscillation
	eliminating evoluiton, filtering, and limiters. It 
	is old and has not been fixed and is not used in
	the code currently. Is just left in case I want
	to steal from it again in the future.
	
## Running

After being compiled using deal.II call:

./[object] [amplitude] [serial] [output diagnostics] [output info and vtu]

amplitude specifies the amplitude of the initial scalar field
the scalar field is initially a gaussian with its center and 
standard deviation specified in the global variables of
scalar_field_engine.cc

serial is just a number included in output diagnostic files.
It is just included so we could run multiple at a time and
this will not affect anything about the simulation

output diagnostics is a boolean (0 or 1) for whether to output
files that are used in the automatic critical search python scripts
we use. Should normally be left at 0 to not create giant 
files.

output info and vtu is a boolean (0 or 1) to decide whether or not to
print information to the command line and output .vtu files
to view in paraview. 
