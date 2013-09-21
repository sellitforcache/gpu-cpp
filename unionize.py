from pyne import ace
from pyne import nucname
import numpy

class cross_section_data:

	def __init__(self):
		self.num_isotopes     = 0
		self.isotope_list     = []
		self.temp_extension   = '.03c'
		self.tables           = []
		self.libraries        = []
		self.main_E_grid      = numpy.array([])
		self.num_main_E		  = 0
		self.reaction_numbers = []
		self.num_reactions    = 0
		self.Ang_grid         = numpy.array([])
		self.Ene_grid  	      = numpy.array([])
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

	def _unionize_main_E_grid(self):

		for table in self.tables:
			self.main_E_grid=numpy.union1d(self.main_E_grid,table.energy)

		self.num_main_E = self.main_E_grid.__len__()


	def _insert_reactions(self):

		for table in self.tables:

			self.reaction_numbers.append(1) #append total abs as MT=1 
			for MT in table.reactions: # reactions is a dict
				self.reaction_numbers.append(MT)

			self.num_reactions=self.reaction_numbers.__len__()

	def _allocate_arrays(self):

		n_columns  = 1 + self.num_isotopes + ( self.num_isotopes + self.num_reactions)  # egrid + totals + (abs + all other reactions (elastic scatter included) )
		n_rows     = self.num_main_E
		self.MT_array = numpy.zeros((n_rows,n_columns),dtype=float,order='C')
		self.MT_array[:,0]=self.main_E_grid


	def _interpolate(self):

		tope_index  = 1
		cum_rxn_len = 0

		for table in self.tables:

			#do total
			this_array = numpy.interp(self.main_E_grid,table.energy,table.sigma_t)
			self.MT_array[:,tope_index]=this_array

			#do abs, start at reaction block
			start_dex=1+self.num_isotopes+cum_rxn_len #(main E grid + total xs block + any previous reaction blocks)
			this_array = numpy.interp(self.main_E_grid,table.energy,table.sigma_a)
			self.MT_array[:,start_dex]=this_array

			for MT in table.reactions: # reactions is a dict
				start_dex=start_dex+1
				cum_rxn_len=cum_rxn_len+1
				rxn = table.reactions[MT]
				IE  = rxn.IE-1
				this_array = numpy.interp(self.main_E_grid,table.energy[IE:],rxn.sigma)
				self.MT_array[:,start_dex]=this_array

			tope_index=tope_index+1

	def _get_MT_array_pointer(self):
		contig.array = numpy.ascontiguousarray(self.MT_array,dtype=numpy.float32)
		return contig_array

	def _get_Egrid_pointer(self):
		E_grid = numpy.ascontiguousarray(self.main_E_grid,dtype=numpy.float32)
		return E_grid

	def _get_Ang_array_pointer(self):
		Ang = numpy.ascontiguousarray(self.Ang_array,dtype=numpy.float32)
		return Ang

	def _get_Ene_array_pointer(self):
		Ene = numpy.ascontiguousarray(self.Ene_array,dtype=numpy.float32)
		return Ene

	def _print_isotopes(self):
		for tope in self.isotope_list:
			print tope


def get_xs_pointer(this_string):

	xs = cross_section_data()
	xs._init_from_string(this_string)
	xs._read_tables()
	xs._unionize_main_E_grid()
	xs._insert_reactions()
	xs._allocate_arrays()
	xs._interpolate()
	return xs._get_array_pointer()

























