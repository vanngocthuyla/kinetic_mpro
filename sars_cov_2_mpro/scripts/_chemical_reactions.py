import jax
import jax.numpy as jnp
import numpy as np

class ChemicalReactions(object):
   """ Computes concentrations of species for a system of chemical equations

   In ceq, equilibrium concentrations are determined by solving the system of 
   chemical equations and conservation equations using an optimization algorithm.
   It is adapted from https://github.com/choderalab/assaytools/blob/818115021fa2cbe11b502908f45fb7d56602b2d4/AssayTools/bindingmodels.py#L243
   In ct, time series are determined from  integrating the rate equations using 
   a ordinary differential equation solver.
   In xi, the extent of each chemical reaction is obtained via a linear least 
   squares regression.
   """
   def __init__(self, chemical_equations, conservation_equations):
      """
      Parameters
      ----------
      chemical_equations : list of dict
          List of chemical_equations, encoded as a dict of stoichiometry of each species
          Example: RL ⇋ R + L is {'RL': -1, 'R' : +1, 'L' : +1}
      conservation_equations : list of dict
          List of mass conservation laws, encoded as a dict of amounts in each species
          Example: [RL] + [R] is {'RL' : +1, 'R' : +1}
      """
      nchemical_equations = len(chemical_equations)
      nconservation_equations = len(conservation_equations)
      nequations = nchemical_equations + nconservation_equations

      # Determine names of all species
      all_species = []
      for chemical_equation in chemical_equations:
        for species in chemical_equation.keys():
          if species not in all_species:
            all_species.append(species)
      all_species = sorted(all_species) # order is now fixed
      nspecies = len(all_species)

      index_of_species = dict([(species,J) for (J,species) in enumerate(all_species)])

      # The stoichiometry matrix helps determine the extent of the reaction
      stoichiometry_matrix = np.zeros((nspecies, nchemical_equations))
      for R in range(nchemical_equations):
        for J in range(nspecies):
          if all_species[J] in chemical_equations[R].keys():
            stoichiometry_matrix[J][R] = chemical_equations[R][all_species[J]]

      # The conservation matrix helps obtain the total concentration
      conservation_matrix = np.zeros((nspecies, nconservation_equations))
      for (R, conservation_equation) in enumerate(conservation_equations):
        for (species, nu) in conservation_equation.items():
          conservation_matrix[index_of_species[species]][R] += nu

      self.all_species = all_species
      self.index_of_species = index_of_species
      self.nspecies = nspecies
      self.chemical_equations = chemical_equations
      self.nchemical_equations = nchemical_equations
      self.conservation_equations = conservation_equations
      self.nconservation_equations = nconservation_equations
      self.stoichiometry_matrix = jnp.array(stoichiometry_matrix)
      self.conservation_matrix = jnp.array(conservation_matrix)

   def logceq_while(self, logKeq, logctot, tol=1.0e-8, max_iter=100):
      """ Equilibrium concentrations

      This is a version of logceq that uses a while loop. 
      It does not work with numpyro.

      Parameters
      ----------
      logKeq : jnp.array
          Log equilibrium constants.
      logctot : jnp.array
          Log total concentrations (log M).
      iters : int
          Number of Gauss-Newton iterations
      Returns
      -------
      logc : numpy.array
          logc[J] is the log concentration of species J
      Examples
      --------
      Simple 1:1 association
      >>> chemical_equations = [ {'RL': -1, 'R' : +1, 'L' : +1} ]
      >>> conservation_equations = [ {'RL' : +1, 'R' : +1}, {'RL' : +1, 'L' : +1} ]
      >>> simple_binding = ChemicalReactions(chemical_equations, conservation_equations)
      >>> log_complex_concentrations = simple_binding.logceq([np.log(1E-6)], [np.log(5E-9), np.log(1E-5)])      
      Competitive 1:1 association
      >>> chemical_equations = [ {'RLa': 1, 'R' : -1, 'La' : -1 }, {'RLb': 1, 'R' : -1, 'Lb' : -1 } ]
      >>> conservation_equations = [ {'RLa' : +1, 'RLb' : +1, 'R' : +1}, {'RLa' : +1, 'La' : +1}, {'RLb' : +1, 'Lb' : +1} ]
      >>> competitive_binding = ChemicalReactions(chemical_equations, conservation_equations)
      >>> logc = competitive_binding.logceq([-np.log(1E-9), -np.log(10E-9)], [np.log(1E-6), np.log(5E-7), np.log(5E-7)])
      """
      # Initial guess
      maxlogc = jnp.max(logctot)
      logci = jnp.array(np.ones((self.nspecies))*maxlogc)

      # Calculate the sum of square deviation and compares to tolerance
      def cond_fun(logc):
        min_logc = jnp.min(logc)
        logsum = jnp.log(jnp.exp(logc - min_logc) @ self.conservation_matrix) \
          + min_logc
        eps = jnp.concatenate((\
          logc @ self.stoichiometry_matrix - logKeq, \
          logsum - logctot))
        ssd = jnp.sum(jnp.square(eps))
        return ssd > tol

      # Gauss-Newton update
      def body_fun(logc):
        min_logc = jnp.min(logc)
        logsum = jnp.log(jnp.exp(logc - min_logc) @ self.conservation_matrix) \
          + min_logc
        eps = jnp.concatenate((\
          logc @ self.stoichiometry_matrix - logKeq, \
          logsum - logctot))
        J = jnp.vstack([self.stoichiometry_matrix.T, 
          self.conservation_matrix.T * jnp.exp(\
            jnp.tile(logc, (self.nconservation_equations,1)) - \
            jnp.tile(logsum, (self.nspecies,1)).T)])
        delta = jnp.linalg.inv(J.T @ J) @ J.T @ eps     
        logc = logc - delta
        logc = jnp.min(jnp.vstack([logc, logci]),0)
        return logc

      # Run Gauss-Newton until acceptable tolerance
      logc = jax.lax.while_loop(cond_fun, body_fun, logci)
      return logc

   def logceq(self, logKeq, logctot, iters=5):
      """ Equilibrium concentrations

      Parameters
      ----------
      logKeq : jnp.array
          Log equilibrium constants.
      logctot : jnp.array
          Log total concentrations (log M).
      iters : int
          Number of Gauss-Newton iterations
      Returns
      -------
      logc : numpy.array
          logc[J] is the log concentration of species J
      Examples
      --------
      Simple 1:1 association
      >>> chemical_equations = [ {'RL': -1, 'R' : +1, 'L' : +1} ]
      >>> conservation_equations = [ {'RL' : +1, 'R' : +1}, {'RL' : +1, 'L' : +1} ]
      >>> simple_binding = ChemicalReactions(chemical_equations, conservation_equations)
      >>> log_complex_concentrations = simple_binding.logceq(np.array([np.log(1E-6)]), np.array([np.log(5E-9), np.log(1E-5)]))
      Competitive 1:1 association
      >>> chemical_equations = [ {'RLa': 1, 'R' : -1, 'La' : -1 }, {'RLb': 1, 'R' : -1, 'Lb' : -1 } ]
      >>> conservation_equations = [ {'RLa' : +1, 'RLb' : +1, 'R' : +1}, {'RLa' : +1, 'La' : +1}, {'RLb' : +1, 'Lb' : +1} ]
      >>> competitive_binding = ChemicalReactions(chemical_equations, conservation_equations)
      >>> logc = competitive_binding.logceq(np.array([-np.log(1E-9), -np.log(10E-9)]), np.array([np.log(1E-6), np.log(5E-7), np.log(5E-7)]))
      """
      # Initial guess
      maxlogc = jnp.max(logctot)
      logci = jnp.array(np.ones((self.nspecies))*maxlogc)
      
      # Gauss-Newton update
      @jax.jit
      def f(logc, xs):
        min_logc = jnp.min(logc)
        logsum = jnp.log(jnp.exp(logc - min_logc) @ self.conservation_matrix) + min_logc
        eps = jnp.concatenate((logc @ self.stoichiometry_matrix - logKeq, logsum - logctot))
        ssd_i = jnp.sum(jnp.square(eps))
        J = jnp.vstack([self.stoichiometry_matrix.T, 
          self.conservation_matrix.T * jnp.exp(
            jnp.tile(logc, (self.nconservation_equations,1)) - \
            jnp.tile(logsum, (self.nspecies,1)).T)])
        delta = jnp.linalg.inv(J.T @ J) @ J.T @ eps     
        logc = logc - delta
        logc = jnp.min(jnp.vstack([logc, logci]),0)
        return (logc, ssd_i)

      # Run Gauss-Newton until acceptable tolerance
      # logc = jax.lax.while_loop(cond_fun, body_fun, logci)
      (logc, ssd) = jax.lax.scan(f, logci, None, length=iters)
      # assert(ssd[-1]<tol)
      return logc

   def ct(self, ci, t, k):
      """ Time series of concentrations

      Parameters
      ----------
      ci : numpy.array
          Initial concentrations of all species
      t : float 
          Times at which concentration is determined
      k : (numpy.array, numpy.array)
          Tuple with forward and reverse rate of each chemical equation

      Returns
      -------
      np.array (TxJ)
          Concentrations of all species (TxJ)
      """
      # from scipy.integrate import odeint
      from jax.experimental.ode import odeint
      return odeint(self.dcdt, jnp.array(ci), t, k)

   def dcdt(self, c, t, k):
      """ Time derivative of concentrations

      Parameters
      ----------
      c : vector
          Concentrations of all species
      t : float 
          Time (not used, but is an argument for the scipy ODE solver)
      k : (numpy.array, numpy.array)
          Tuple with forward and reverse rate of each chemical equation

      Returns
      -------
      np.array
          Time derivatives of concentrations
      """
      v = self.dxidt(c, t, k)
      return jnp.matmul(self.stoichiometry_matrix, v)

   def xi(self, c, ci):
      """ Progress of the reactions

      Parameters
      ----------
      c : numpy.array
          c[J] is the concentration of species J
      ci : numpy.array
          ci[J] is the initial concentration of species J

      Returns
      -------
      np.array
          progress of each reaction
      """
      return jnp.linalg.lstsq(self.stoichiometry_matrix, c-ci, rcond=None)[0]

   def dxidt(self, c, k):
      """ Time derivative of reaction progress

      Parameters
      ----------
      c : vector
          Concentrations of all species
      k : (numpy.array, numpy.array)
          Tuple with forward and reverse rate of each chemical equation

      Returns
      -------
      np.array
          Time derivatives of the progress variable xi

      # TODO: Time dependence?
      """
      (kf, kr) = k
      v = jnp.zeros(self.nchemical_equations)
      for R in range(self.nchemical_equations):
        vf = kf[R]
        vr = kr[R]
        for J in range(self.nspecies):
          is_reactant = (self.stoichiometry_matrix[J][R] < 0.)
          is_product = (self.stoichiometry_matrix[J][R] > 0.)
          vf = vf*(c[J]**(is_reactant*-self.stoichiometry_matrix[J][R]))
          vr = vr*(c[J]**(is_product*self.stoichiometry_matrix[J][R]))
        # v[R] = vf - vr
        v = jax.ops.index_update(v, R, vf - vr)
      return v

   def species_dict_to_array(self, species_dict):
      """ Convert dictionary with species labels into an array
      
      Follows the order in self.all_species

      Parameters
      ----------
      species_dict : dict of str : float
          species_dict[species] is the value for a species

      Returns
      -------
      np.array
          Values in the standardized order for species
      """
      X = np.zeros((self.nspecies))
      for (J, species) in enumerate(self.all_species):
        if species in species_dict.keys():
          X[J] = species_dict[species]
      return X