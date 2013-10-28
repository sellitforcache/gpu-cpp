import pylab
import sys
import os
import re

def get_serpent_det(filepath):
	fobj    = open(filepath)
	fstr    = fobj.read()
	names   = re.findall('[a-zA-Z]+ *= *\[',fstr)
	data    = re.findall('\[ *\n[\w\s+-.]+\];',fstr)
	alldata = dict()
	dex     = 0
	for name in names:
		varname  = name.split()[0]
		moredata = re.findall(' [ .+-eE0-9^\[]+\n',data[dex])
		thisarray = numpy.array([])
		for lines in moredata:
			thisarray=numpy.append(thisarray,numpy.array(moredata[1].split(),dtype=float),0)
		alldata[varname]=thisarray
		dex = dex + 1
	return alldata


if   case=='water':
	tally      = numpy.loadtxt('water.tally')
	tallybins  = numpy.loadtxt('water.tallybins')
	serpdata   = get_serpent_det('..serpent-benchmark/nonfiss_mono2_det0.m')
elif case== 'carbon':
	tally      = numpy.loadtxt('carbon.tally')
	tallybins  = numpy.loadtxt('carbon.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/carbon_mono2_det0.m')
elif case== 'lithium':
	tally      = numpy.loadtxt('lithium.tally')
	tallybins  = numpy.loadtxt('lithium.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/lithium_mono2_det0.m')
elif case== 'hydrogen':
	tally      = numpy.loadtxt('hydrogen.tally')
	tallybins  = numpy.loadtxt('hydrogen.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/hydrogen_mono2_det0.m')
elif case== 'hydrogen2':
	tally      = numpy.loadtxt('hydrogen2.tally')
	tallybins  = numpy.loadtxt('hydrogen2.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/hydrogen2_mono2_det0.m')

plot_spectra_serpent(DETfluxlogE,DETfluxlog)
plot_spectra_warp(tally,tallybins,'r')

set(gca,'FontSize',14)
title({'Serpent2 (Serial) vs. WARP','4e6 histories, 2MeV point source at origin of 84x84x84cm water block'})
xlabel('Energy (MeV)')
ylabel('Normalized Flux/Lethary')
legend('Serpent2, 18.2 minutes','WARP, 5.76 minutes','MCNP6.1','Location','NW')
grid on
a=ylim();
ylim(a*.5)
xlim([2e-11,25])