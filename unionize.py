from pyne import ace
from pyne import nucname
import numpy
import sys

class cross_section_data:

	def __init__(self):
		self.num_isotopes     = 0
		self.isotope_list     = []
		self.temp_extension   = '.03c'
		self.tables           = []
		self.libraries        = []
		self.awr 			  = []
		self.Q 				  = []
		self.num_main_E		  = 0
		self.reaction_numbers = []
		self.reaction_numbers_total = []
		self.num_reactions    = 0
		self.MT_E_grid        = numpy.array([],dtype=numpy.float32,order='C')
		self.MT_array		  = numpy.array([],dtype=numpy.float32,order='C')

	def _init_from_string(self,this_string):
		self.isotope_list = this_string.split(',')

	def _add_isotope(self,  isotope):
		self.isotope_list.append(isotope)

	def _read_tables(self):

		for tope in self.isotope_list:
			tope_number = nucname.mcnp(tope)
			#print 'trying to load '+str(tope_number)
			self.libraries.append(ace.Library('/usr/local/SERPENT/xsdata/endfb7/acedata/'+str(tope_number)+'ENDF7.ace'))

		for lib in self.libraries:
			lib.read()
			iname=lib.tables.keys()[0][0:-4]   #strip off temp to get isotope name
			print "Loading "+iname+self.temp_extension
			self.tables.append(lib.find_table(iname+self.temp_extension))

		self.num_isotopes=self.libraries.__len__()

	def _unionize(self):

		for table in self.tables:
			self.MT_E_grid=numpy.union1d(self.MT_E_grid,table.energy)
			# unionize the scattering energies in as well!  if present of course
			for MT in table.reactions:
				rxn = table.reactions[MT]
				if hasattr(rxn,"ang_energy_in"):
					self.MT_E_grid=numpy.union1d(self.MT_E_grid,rxn.ang_energy_in)

		self.num_main_E   = self.MT_E_grid.__len__()
		#print self.MT_E_grid.shape
		#print self.MT_E_grid

	pass
	def _insert_reactions(self):

		
		for table in self.tables:
			#append ones to front
			self.reaction_numbers.append(1)
			self.Q.append(0)
			#append this topes AWR
			self.awr.append(table.awr)
			#append totals
			self.reaction_numbers_total.append(table.reactions.__len__())

		#append reaction numbers
		for table in self.tables:
			for MT in table.reactions: # reactions is a dict
				rxn = table.reactions[MT]
				self.reaction_numbers.append(MT)
				self.Q.append(rxn.Q)
				self.num_reactions += 1

		#print self.num_reactions
		print self.reaction_numbers
		print self.Q
		#print self.reaction_numbers_total

	def _allocate_arrays(self):

		n_columns  = self.num_isotopes + self.num_reactions  # totals + ( all other reactions (elastic scatter included) )
		n_rows     = self.num_main_E

		self.MT_array  = numpy.zeros((n_rows,n_columns),dtype=float,order='C')

	def _interpolate(self):

		tope_index  = 0
		MT_array_dex  = self.num_isotopes  #(total xs block + any previous reaction blocks)

		for table in self.tables:

			#print "interpolating isotope "+str(tope_index)

			#do this isotopes entry in the total block
			this_array = numpy.interp( self.MT_E_grid, table.energy, table.sigma_t , left=0.0 )
			self.MT_array[:,tope_index]=this_array

			#do abs and higher, start at reaction block
			#this_array    = numpy.interp( self.MT_E_grid, table.energy, table.sigma_a ) #total abs vector
			#self.MT_array[:,MT_array_dex] = this_array
			#MT_array_dex = MT_array_dex + 1  #increment MT array index

			for MT in table.reactions:
				rxn        = table.reactions[MT]
				IE         = rxn.IE - 1       #convert to python/C indexing 
				this_array = numpy.interp( self.MT_E_grid, table.energy[IE:], rxn.sigma , left=0.0 )  #interpolate MT cross section
				self.MT_array[:,MT_array_dex] = this_array  # insert into the MT array

				#  this MT is done, increment counter
				MT_array_dex = MT_array_dex +1

			#this isotope is done, increment counter
			tope_index  = tope_index+1

	def _get_MT_numbers_pointer(self):
		MT_num_array = numpy.ascontiguousarray(numpy.array(self.reaction_numbers,order='C'),dtype=numpy.uint32)
		return MT_num_array

	def _get_awr_pointer(self):
		awr_array = numpy.ascontiguousarray(numpy.array(self.awr,order='C'),dtype=numpy.float32)
		return awr_array

	def _get_Q_pointer(self):
		awr_array = numpy.ascontiguousarray(numpy.array(self.Q,order='C'),dtype=numpy.float32)
		return awr_array

	def _get_MT_array_pointer(self):
		self.MT_array = numpy.ascontiguousarray(self.MT_array,dtype=numpy.float32)
		return self.MT_array

	def _get_main_Egrid_pointer(self):
		E_grid = numpy.ascontiguousarray(self.MT_E_grid,dtype=numpy.float32)
		return E_grid

	def _get_length_numbers_pointer(self):
		lengths = numpy.ascontiguousarray( numpy.array([self.num_isotopes, self.num_main_E, self.num_reactions], order='C') ,dtype=numpy.uint32)
		return lengths

	def _get_MT_numbers_total_pointer(self):
		numbers = numpy.array(self.reaction_numbers_total,order='C')
		numbers = numpy.cumsum(numbers)
		numbers = numpy.ascontiguousarray(numbers,dtype=numpy.uint32)
		return numbers

	def _print_isotopes(self):
		for tope in self.isotope_list:
			print tope

	def _get_scattering_data(self,row,col):
		# scatter table returned in this form
		# returns [nextDex, length, mu, cdf] if scattering data exists

		#find the isotope we are in
		numbers = numpy.cumsum(self.reaction_numbers_total)
		isotope = 0
		for n in numbers:
			if (col - self.num_isotopes) <= n:
				break
			else:
				isotope = isotope + 1

		table = self.tables[isotope]
		MTnum = self.reaction_numbers[col]
		rxn   = table.reactions[MTnum]
		#print "here now"
		if hasattr(rxn,"ang_energy_in"):
			#print "isotope "+str(isotope)+", MT = "+str(MTnum)+" has scattering data"
			scatterE   = rxn.ang_energy_in
			scatterMu  = rxn.ang_cos 
			scatterCDF = rxn.ang_cdf
			# check length
			assert scatterE.__len__() > 0
			# get the energy from this index
			this_E = self.MT_E_grid[row]
			# find the index of the scattering table energy
			if this_E >= scatterE[0]:
				scatter_dex = numpy.where( scatterE >= this_E )[0][0]
				#get energy of next bin
				if scatter_dex == scatterE.__len__()-1:
					next_E = self.MT_E_grid[-1]
				else:
					next_E = scatterE[scatter_dex+1]
				# find main E grid indext of next energy
				nextDex = numpy.where( self.MT_E_grid == next_E )[0][0]
				print "MT = "+str(MTnum)
				print row,col
				print this_E,next_E
				print scatter_dex
				print nextDex
				print scatterE
				# construct vector
				vlen  = scatterCDF[scatter_dex].__len__()
				cdf   = numpy.ascontiguousarray(scatterCDF[scatter_dex],dtype=numpy.float32)  # C/F order doesn't matter for 1d arrays
				mu    = numpy.ascontiguousarray(scatterMu[scatter_dex], dtype=numpy.float32)
				#check to make sure the same length
				assert vlen == mu.__len__()
				# return
				return [nextDex,vlen,mu,cdf]
			else:  # return 0 if below the first energy]
				next_E = scatterE[0]
				nextDex = numpy.where( self.MT_E_grid == next_E )[0][0]
				return [nextDex,0,numpy.array([0]),numpy.array([0])]
		else:
			print "isotope "+str(isotope)+", MT = "+str(MTnum)+" has no angular tables"
			nextE   = self.MT_E_grid[self.num_main_E-1]
			nextDex = self.MT_E_grid.__len__()
			return [nextDex,0,numpy.array([0]),numpy.array([0])]






	def _get_energy_data(self,isotope,MTnum,energy):
		# scatter table returned in this form
		# returns [nextE, length, mu, cdf]

		#print isotope,MTnum,energy

		table = self.tables[isotope]
		rxn   = table.reactions[MTnum]
		#print "here now"
		if hasattr(rxn,"energy_dist"):
			print "LAW="+str(rxn.energy_dist.law)+" MT="+str(MTnum)+" E="+str(energy)
			if rxn.energy_dist.law == 3:
				nextE = self.MT_E_grid[self.num_main_E-1]
				return [nextE,0,3,numpy.array([0]),numpy.array([0])]
			else:
				scatterE   = rxn.energy_dist.energy_in
				scatterMu  = rxn.energy_dist.energy_out
				scatterCDF = rxn.energy_dist.cdf
				law        = rxn.energy_dist.law
				#print "lengths = "+str(scatterMu.__len__())
				#print scatterMu[1]
				#print scatterCDF[1]
		else:
			#print "no angular tables"
			nextE = self.MT_E_grid[self.num_main_E-1]
			return [nextE,0,0,numpy.array([0]),numpy.array([0])]


		# check length
		assert scatterE.__len__() > 0

		# find the proper energy index
		# "snap to grid" method
		dex = numpy.argmax( scatterE >= energy )

		#print scatterE
		#print energy
		#print scatterE >= energy
		#print "dex = "+str(dex)

		#get energy of next bin
		if dex == scatterE.__len__()-1:
			nextE = scatterE  [dex]
		else:
			nextE = scatterE  [dex+1]

		# return 0 if below the first energy
		if energy < scatterE[0]:
			return [nextE,0,0,numpy.array([0]),numpy.array([0])]

		# construct vector
		vlen  = scatterCDF[dex].__len__()
		cdf   = numpy.ascontiguousarray(scatterCDF[dex],dtype=numpy.float32)  # C/F order doesn't matter for 1d arrays
		mu    = numpy.ascontiguousarray(scatterMu[dex], dtype=numpy.float32)

		#check to make sure the same length
		assert vlen == mu.__len__()

		# return
		return [nextE,vlen,law,mu,cdf]

	def _get_nu_data():
		#find the isotope we are in
		numbers = numpy.cumsum(self.reaction_numbers_total)
		isotope = 0
		for n in numbers:
			if (col - self.num_isotopes) <= n:
				break
			else:
				isotope = isotope + 1

		table = self.tables[isotope]
		MTnum = self.reaction_numbers[col]
		rxn   = table.reactions[MTnum]

		if hasattr(table,"nu_t_type"):
		#print "isotope "+str(isotope)+", MT = "+str(MTnum)+" has nu type"+table.nu_t_type
		#return the nu value for this isotope
			nu = numpy.interp( self.MT_E_grid, table.nu_t_energy, table.nu_t_value , left=0.0 ) 
			nextDex = self.MT_E_grid.__len__() + 1
			return [nextDex,-1,nu,nu]

























