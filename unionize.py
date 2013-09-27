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

		self.num_main_E   = self.MT_E_grid.__len__()


	def _insert_reactions(self):

		for table in self.tables:

			self.reaction_numbers.append(1) #append total abs as MT=1 
			self.reaction_numbers_total.append(table.reactions.__len__())
			for MT in table.reactions: # reactions is a dict
				self.reaction_numbers.append(MT)

		self.num_reactions=self.reaction_numbers.__len__()

		print self.num_reactions

	def _allocate_arrays(self):

		n_columns  = 2*self.num_isotopes + self.num_reactions  # totals + (abs + all other reactions (elastic scatter included) )
		n_rows     = self.num_main_E

		self.MT_array  = numpy.zeros((n_rows,n_columns),dtype=float,order='C')

	def _interpolate(self):

		tope_index  = 0
		MT_array_dex  = self.num_isotopes  #(total xs block + any previous reaction blocks)

		for table in self.tables:

			#print "interpolating isotope "+str(tope_index)

			#do this isotopes entry in the total block
			this_array = numpy.interp( self.MT_E_grid, table.energy, table.sigma_t )
			self.MT_array[:,tope_index]=this_array

			#do abs and higher, start at reaction block
			this_array    = numpy.interp( self.MT_E_grid, table.energy, table.sigma_a ) #total abs vector
			self.MT_array[:,MT_array_dex] = this_array
			MT_array_dex = MT_array_dex + 1  #increment MT array index

			for MT in table.reactions:
				rxn        = table.reactions[MT]
				IE         = rxn.IE - 1       #convert to python/C indexing 
				this_array = numpy.interp( self.MT_E_grid, table.energy[IE:], rxn.sigma )  #interpolate MT cross section
				self.MT_array[:,MT_array_dex] = this_array  # insert into the MT array

				#  this MT is done, increment counter
				MT_array_dex = MT_array_dex +1

			#this isotope is done, increment counter
			tope_index  = tope_index+1

	def _get_MT_number_pointer(self):
		MT_num_array = numpy.ascontiguousarray(numpy.array(self.reaction_numbers,order='C'),dtype=numpy.uint32)
		return MT_num_array

	def _get_MT_array_pointer(self):
		self.MT_array = numpy.ascontiguousarray(self.MT_array,dtype=numpy.float32)
		return self.MT_array

	def _get_main_Egrid_pointer(self):
		E_grid = numpy.ascontiguousarray(self.MT_E_grid,dtype=numpy.float32)
		return E_grid

	def _get_length_numbers_pointer(self):
		lengths = numpy.ascontiguousarray( numpy.array([self.num_isotopes, self.num_main_E, self.num_reactions, self.num_matrix_E, self.num_ang_cos, self.num_ene_E], order='C') ,dtype=numpy.uint32)
		return lengths

	def _get_MT_number_totals_pointer(self):
		numbers = numpy.ascontiguousarray(numpy.array(self.reaction_numbers_total,order='C'),dtype=numpy.uint32)
		return numbers

	def _print_isotopes(self):
		for tope in self.isotope_list:
			print tope

	def _get_scattering_data(self,isotope,MTnum,energy):
		# scatter table returned in this form
		# returns [nextE, length, mu, cdf]

		table = self.tables[isotope]
		rxn   = table.reactions[MTnum]
		scatterE   = rxn.ang_energy_in
		scatterMu  = rxn.ang_cos 
		scatterCDF = rxn.ang_cdf

		# return 0 if below the first energy
		if energy < scatterE[0]:
			return numpy.array([0])

		# find the proper energy index
		# "snap to grid" method
		dex = numpy.argmax( scatterE >= energy )

		# construct vector
		nextE = scatterE  [dex+1]
		vlen  = scatterCDF[dex].__len__()
		cdf   = numpy.ascontiguousarray(scatterCDF[dex],dtype=numpy.float32)  # C/F order doesn't matter for 1d arrays
		mu    = numpy.ascontiguousarray(scatterMu[dex],dtype=numpy.float32)

		#check to make sure the same length
		assert vlen == mu.__len__()

		# return
		return [nextE,vlen,mu,cdf]

























