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
		self.Ang_grid         = numpy.array([],dtype=numpy.float32,order='C')
		self.Ene_grid  	      = numpy.array([],dtype=numpy.float32,order='C')
		self.matrix_E_grid    = numpy.array([],dtype=numpy.float32,order='C')
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
				print MT
				if hasattr(rxn, 'ang_energy_in'):
					self.matrix_E_grid=numpy.union1d(self.matrix_E_grid,rxn.ang_energy_in)  # add angular energy_in energy points
					for energy_dex in range(rxn.ang_energy_in.__len__()):
						self.Ang_grid = numpy.union1d(self.Ang_grid,rxn.ang_cos[energy_dex])
				if hasattr(rxn, 'energy_dist'):
					energy_dist=rxn.energy_dist
					if hasattr(energy_dist, 'energy_in'):
						self.matrix_E_grid=numpy.union1d(self.matrix_E_grid,rxn.energy_dist.energy_in)  # add energy_in points 
						for energy_dex in range(rxn.energy_dist.energy_in.__len__()):
							self.Ene_grid = numpy.union1d(self.Ene_grid,rxn.energy_dist.energy_out[energy_dex])
		self.num_main_E = self.MT_E_grid.__len__()
		self.num_ang_cos  = self.Ang_grid.__len__()
		self.num_ene_E  = self.Ene_grid.__len__()
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
		self.Ene_array = numpy.zeros((self.num_matrix_E,self.num_ene_E,self.num_reactions),dtype=float,order='C')


	def _interpolate(self):

		tope_index  = 1
		cum_rxn_len = 0

		for table in self.tables:

			#do total
			this_array = numpy.interp(self.MT_E_grid,table.energy,table.sigma_t)
			self.MT_array[:,tope_index]=this_array

			#do abs, start at reaction block
			start_dex=1+self.num_isotopes+cum_rxn_len #(main E grid + total xs block + any previous reaction blocks)
			this_array = numpy.interp(self.MT_E_grid,table.energy,table.sigma_a)
			self.MT_array[:,start_dex]=this_array

			for MT in table.reactions: # reactions is a dict
				print "interpolating MT "+str(MT)
				start_dex=start_dex+1
				cum_rxn_len=cum_rxn_len+1
				rxn = table.reactions[MT]
				IE  = rxn.IE-1
				this_array_MT = numpy.interp( self.MT_E_grid, table.energy[IE:],  rxn.sigma)
				self.MT_array[:,start_dex]=this_array_MT   # insert reaction into MT table
				if hasattr(rxn, 'energy_dist'):
					for energy_dex in range(rxn.energy_dist.energy_in.__len__()):
						this_array_ene= numpy.interp( self.Ene_grid,    rxn.energy_dist.energy_out[energy_dex],  rxn.energy_dist.cdf[energy_dex])  # horizontal interpolation 
						#set as constant within the energy range, ie only horizontal interpolation, no vertical
						#find index
						dex     = numpy.argmax(rxn.energy_dist.energy_in[energy_dex]>=self.MT_E_grid)
						if dex<start_dex:
							sys.exit("inital energy distribution is below threshold")
						if energy_dex == (rxn.energy_dist.energy_in.__len__()-1):
							nextdex = self.num_main_E-1
						else:
							nextdex = numpy.argmax(rxn.energy_dist.energy_in[energy_dex+1]>=self.MT_E_grid)
						for  this_dex in range(dex,nextdex):
							self.Ene_array[this_dex,:,start_dex]=this_array_ene
				#else:
				#	self.Ene_array[this_dex,:,start_dex]=numpy.zeros((self.num_ene_E,num_matrix_E),dtype=float,order='C')
				if hasattr(rxn, 'ang_energy_in'):
					for energy_dex in range(rxn.ang_energy_in.__len__()):
						this_array_ang= numpy.interp( self.Ang_grid,    rxn.ang_cos[energy_dex],  rxn.ang_cdf[energy_dex])
						#set as constant within the energy range, ie only horizontal interpolation, no vertical
						#find index
						dex     = numpy.argmax(rxn.ang_energy_in[energy_dex]>=self.MT_E_grid)
						if dex<start_dex:
							sys.exit("inital angular distribution is below threshold")
						if energy_dex == (rxn.ang_energy_in.__len__()-1):
							nextdex = self.num_main_E-1
						else:
							nextdex = numpy.argmax(rxn.ang_energy_in[energy_dex+1]>=self.MT_E_grid)
						for  this_dex in range(dex,nextdex):
							self.Ang_array[this_dex,:,start_dex]=this_array_ang	
				#else:  #should be zero anyway
			    #	self.Ene_array[this_dex,:,start_dex]=numpy.zeros((self.num_ang_cos,num_matrix_E),dtype=float,order='C')
			tope_index=tope_index+1

	def _get_MT_number_pointer(self):
		MT_num_array = numpy.ascontiguousarray(numpy.array(self.reaction_numbers),dtype=numpy.uint32,order='C')
		return MT_num_array

	def _get_MT_array_pointer(self):
		MT_array = numpy.ascontiguousarray(self.MT_array,dtype=numpy.float32,order='C')
		return MT_array

	def _get_main_Egrid_pointer(self):
		E_grid = numpy.ascontiguousarray(self.MT_E_grid,dtype=numpy.float32,order='C')
		return E_grid

	def _get_Ang_array_pointer(self):
		Ang = numpy.ascontiguousarray(self.Ang_array,dtype=numpy.float32,order='C')
		return Ang

	def _get_Ang_Egrid_pointer(self):
		Ang_grid = numpy.ascontiguousarray(self.Ang_grid,dtype=numpy.float32,order='C')
		return Ang_grid

	def _get_Ene_array_pointer(self):
		Ene = numpy.ascontiguousarray(self.Ene_array,dtype=numpy.float32,order='C')
		return Ene

	def _get_Ene_Egrid_pointer(self):
		Ene_grid = numpy.ascontiguousarray(self.Ene_grid,dtype=numpy.float32,order='C')
		return Ene_grid

	def _get_length_numbers_pointer(self):
		lengths = numpy.ascontiguousarray(numpy.array([self.num_isotopes, self.num_main_E, self.num_ang_cos, self.num_ene_E]),dtype=numpy.uint32,order='C')
		return lengths

	def _get_MT_number_totals_pointer(self):
		numbers = numpy.ascontiguousarray(numpy.array(self.reaction_numbers_total),dtype=numpy.uint32,order='C')
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
			xs._get_Ang_Egrid_pointer(), \
			xs._get_Ang_array_pointer(), \
			xs._get_Ene_Egrid_pointer(), \
			xs._get_Ene_array_pointer()]





