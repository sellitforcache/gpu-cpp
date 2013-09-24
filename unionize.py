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
		self.num_matrix_E     = 0
		self.num_ang_cos      = 0
		self.num_ene_E        = 0
		self.reaction_numbers = []
		self.reaction_numbers_total = []
		self.num_reactions    = 0
		self.MT_E_grid        = numpy.array([],dtype=numpy.float32,order='C')
		self.matrix_E_grid    = numpy.array([],dtype=numpy.float32,order='C')
		self.Ang_cos_grid	  = numpy.array([],dtype=numpy.float32,order='C')
		self.Ene_E_grid     = numpy.array([],dtype=numpy.float32,order='C')
		self.MT_array		  = numpy.array([],dtype=numpy.float32,order='C')
		self.Ang_array		  = numpy.array([],dtype=numpy.float32,order='C')
		self.Ene_array		  = numpy.array([],dtype=numpy.float32,order='C')

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
			for MT in table.reactions: # reactions is a dict
				rxn = table.reactions[MT]
				if hasattr(rxn, 'ang_energy_in'):
					self.matrix_E_grid=numpy.union1d(self.matrix_E_grid,rxn.ang_energy_in)  # add angular energy_in energy points
					for energy_dex in range(rxn.ang_energy_in.__len__()):
						self.Ang_cos_grid = numpy.union1d(self.Ang_cos_grid,rxn.ang_cos[energy_dex])
				if hasattr(rxn, 'energy_dist'):
					energy_dist=rxn.energy_dist
					if hasattr(energy_dist, 'energy_in'):
						self.matrix_E_grid=numpy.union1d(self.matrix_E_grid,rxn.energy_dist.energy_in)  # add energy_in points 
						for energy_dex in range(rxn.energy_dist.energy_in.__len__()):
							self.Ene_E_grid = numpy.union1d(self.Ene_E_grid,rxn.energy_dist.energy_out[energy_dex])

		self.num_main_E   = self.MT_E_grid.__len__()
		self.num_ang_cos  = self.Ang_cos_grid.__len__()
		self.num_ene_E    = self.Ene_E_grid.__len__()
		self.num_matrix_E = self.matrix_E_grid.__len__()


	def _insert_reactions(self):

		for table in self.tables:

			self.reaction_numbers.append(1) #append total abs as MT=1 
			self.reaction_numbers_total.append(table.reactions.__len__())
			for MT in table.reactions: # reactions is a dict
				self.reaction_numbers.append(MT)

			self.num_reactions=self.reaction_numbers.__len__()

	def _allocate_arrays(self):

		print "allocating arrays"

		n_columns  = self.num_isotopes + ( self.num_isotopes + self.num_reactions)  # totals + (abs + all other reactions (elastic scatter included) )
		n_rows     = self.num_main_E

		print n_rows,n_columns
		print self.num_matrix_E,self.num_ang_cos,self.num_ene_E

		self.MT_array  = numpy.zeros((n_rows,n_columns),dtype=float,order='C')
		self.Ang_array = numpy.zeros((self.num_matrix_E,self.num_ang_cos,self.num_reactions),dtype=float,order='C')
		self.Ene_array = numpy.zeros((self.num_matrix_E,self.num_ene_E  ,self.num_reactions),dtype=float,order='C')


	def _interpolate(self):

		tope_index  = 0
		MT_array_dex  = self.num_isotopes  #(total xs block + any previous reaction blocks)

		for table in self.tables:

			print "interpolating isotope "+str(tope_index)

			#do this isotopes entry in the total block
			this_array = numpy.interp( self.MT_E_grid, table.energy, table.sigma_t )
			self.MT_array[:,tope_index]=this_array

			#do abs and higher, start at reaction block
			this_array    = numpy.interp( self.MT_E_grid, table.energy, table.sigma_a ) #total abs vector
			self.MT_array[:,MT_array_dex] = this_array
			MT_array_dex = MT_array_dex + 1  #increment MT array index

			for MT in table.reactions:
				print "   interpolating MT "+str(MT)
				#
				#	interpolate MT cross sections 
				#
				rxn        = table.reactions[MT]
				IE         = rxn.IE - 1       #convert to python/C indexing 
				this_array = numpy.interp( self.MT_E_grid, table.energy[IE:], rxn.sigma )  #interpolate MT cross section
				self.MT_array[:,MT_array_dex] = this_array  # insert into the MT array
				
				#
				#   interpolate scattering matrix angular distributions for this MT
				#
				if hasattr(rxn,'ang_energy_in'):
					dex_list=[]
					individual_dex=0
					for this_E in rxn.ang_energy_in :
						# get matrix E grid index for this cdf, can use == since it HAS TO BE in the vector, that's the point of unionizing
						dex = numpy.argmax( this_E == self.matrix_E_grid )
						dex_list.append(dex)
						this_array = numpy.interp( self.Ang_cos_grid, rxn.ang_cos[individual_dex], rxn.ang_cdf[individual_dex] )
						self.Ang_array[dex,:,MT_array_dex] = this_array   #insert into array at dex, leave interpolation in between until after all insertions
						individual_dex=individual_dex+1
					if dex_list[dex_list.__len__()-1] != self.num_matrix_E:
						dex_list.append(self.num_matrix_E)  # append the total length to the list if not present
					for j in range(dex_list.__len__()-1):    #copy upward between indicies
						for k in range(dex_list[j],dex_list[j+1]):
							self.Ang_array[k,:,MT_array_dex] = self.Ang_array[j,:,MT_array_dex]

				#
				#   interpolate scattering matrix energylar distributions for this MT
				#
				if hasattr(rxn,'energy_dist'):
					dex_list=[]
					individual_dex=0
					if rxn.energy_dist.law==3:
						energy_in=[rxn.energy_dist.energy[0]] # set energy as threshold
					else:
						energy_in = rxn.energy_dist.energy_in
					for this_E in energy_in :
						# get matrix E grid index for this cdf, can use == since it HAS TO BE in the vector, that's the point of unionizing
						dex = numpy.argmax( this_E >= self.matrix_E_grid )
						dex_list.append(dex)
						if rxn.energy_dist.law==3:  #set as a bunch of -3's
							this_array = numpy.multiply(numpy.ones((1,self.num_ene_E),dtype=numpy.float32,order='C'),(-3.0))
						else:  #interpolate cdf
							this_array = numpy.interp( self.Ene_E_grid, rxn.energy_dist.energy_out[individual_dex], rxn.energy_dist.cdf[individual_dex] )
						self.Ene_array[dex,:,MT_array_dex] = this_array   #insert into array at dex, leave interpolation in between until after all insertions
						individual_dex=individual_dex+1
					if dex_list[dex_list.__len__()-1] != self.num_matrix_E:
						dex_list.append(self.num_matrix_E)  # append the total length to the list if not present
					for j in range(dex_list.__len__()-1):    #copy upward between indicies
						for k in range(dex_list[j],dex_list[j+1]):
							self.Ene_array[k,:,MT_array_dex] = self.Ene_array[j,:,MT_array_dex]

				#  this MT is done, increment counter
				MT_array_dex = MT_array_dex +1

			#this isotope is done, increment counter
			tope_index  = tope_index+1

	def _get_MT_number_pointer(self):
		MT_num_array = numpy.ascontiguousarray(numpy.array(self.reaction_numbers,order='C'),dtype=numpy.uint32)
		return MT_num_array

	def _get_MT_array_pointer(self):
		MT_array = numpy.ascontiguousarray(self.MT_array,dtype=numpy.float32)
		return MT_array

	def _get_main_Egrid_pointer(self):
		E_grid = numpy.ascontiguousarray(self.MT_E_grid,dtype=numpy.float32)
		return E_grid

	def _get_Ang_array_pointer(self):
		Ang = numpy.ascontiguousarray(self.Ang_array,dtype=numpy.float32)
		return Ang

	def _get_Ang_Egrid_pointer(self):
		Ang_grid = numpy.ascontiguousarray(self.Ang_cos_grid,dtype=numpy.float32)
		return Ang_grid

	def _get_Ene_array_pointer(self):
		Ene = numpy.ascontiguousarray(self.Ene_array,dtype=numpy.float32)
		return Ene

	def _get_Ene_Egrid_pointer(self):
		Ene_grid = numpy.ascontiguousarray(self.Ene_E_grid,dtype=numpy.float32)
		return Ene_grid

	def _get_Matrix_Egrid_pointer(self):
		matrix_grid = numpy.ascontiguousarray(self.matrix_E_grid,dtype=numpy.float32)
		return matrix_grid

	def _get_length_numbers_pointer(self):
		lengths = numpy.ascontiguousarray( numpy.array([self.num_isotopes, self.num_main_E, self.num_matrix_E, self.num_ang_cos, self.num_ene_E], order='C') ,dtype=numpy.uint32)
		return lengths

	def _get_MT_number_totals_pointer(self):
		numbers = numpy.ascontiguousarray(numpy.array(self.reaction_numbers_total,order='C'),dtype=numpy.uint32)
		return numbers

	def _print_isotopes(self):
		for tope in self.isotope_list:
			print tope


def get_xs_pointers(this_string):

	xs = cross_section_data()
	xs._init_from_string(this_string)
	xs._read_tables()
	xs._unionize()
	xs._insert_reactions()
	xs._allocate_arrays()
	xs._interpolate()
	return [xs._get_length_numbers_pointer(), \
			xs._get_MT_number_totals_pointer(), \
			xs._get_MT_number_pointer(), \
			xs._get_main_Egrid_pointer(), \
			xs._get_MT_array_pointer(), \
			xs._get_Matrix_Egrid_pointer(), \
			xs._get_Ang_Egrid_pointer(), \
			xs._get_Ang_array_pointer(), \
			xs._get_Ene_Egrid_pointer(), \
			xs._get_Ene_array_pointer()]





