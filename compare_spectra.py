#! /usr/bin/python
import pylab
import sys
import numpy
import os
import re

case = sys.argv[1]

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
		thisarray = numpy.array(moredata[0].split(),dtype=float)
		for line in moredata[1:]:
			thisarray=numpy.vstack((thisarray,numpy.array(line.split(),dtype=float)))
		alldata[varname]=numpy.mat(thisarray)
		dex = dex + 1
	return alldata


if   case=='water':
	tally      = numpy.loadtxt('water.tally')
	tallybins  = numpy.loadtxt('water.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/nonfiss_mono2_det0.m')
elif case=='isowater':
	tally      = numpy.loadtxt('water.tally')
	tallybins  = numpy.loadtxt('water.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/nonfiss_mono2_iso_det0.m')
elif case== 'carbon':
	tally      = numpy.loadtxt('carbon.tally')
	tallybins  = numpy.loadtxt('carbon.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/carbon_mono2_det0.m')
elif case== 'lithium':
	tally      = numpy.loadtxt('lithium.tally')
	tallybins  = numpy.loadtxt('lithium.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/lithium_mono2_det0.m')
elif case== 'hydrogen1':
	tally      = numpy.loadtxt('hydrogen1.tally')
	tallybins  = numpy.loadtxt('hydrogen1.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/hydrogen_mono2_det0.m')
elif case== 'hydrogen2':
	tally      = numpy.loadtxt('hydrogen2.tally')
	tallybins  = numpy.loadtxt('hydrogen2.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/hydrogen2_mono2_det0.m')


widths=numpy.diff(tallybins);
avg=(tallybins[:-1]+tallybins[1:])/2;
newflux=numpy.divide(tally[:-1,0],widths)
newflux=numpy.multiply(newflux,avg)
newflux=numpy.divide(newflux,40e5)
newflux=numpy.divide(newflux,numpy.max(newflux))

serpE=serpdata['DETfluxlogE'][:,2]
serpF=serpdata['DETfluxlog'][:,10]
serpF=numpy.divide(serpdata['DETfluxlog'][:,10],numpy.max(serpdata['DETfluxlog'][:,10]))


p1=pylab.semilogx(serpE,serpF,'b',avg,newflux,'r',linestyle='steps-mid')
pylab.xlabel('Energy (MeV)')
pylab.ylabel('Normalized Flux/Lethary')
pylab.title('Serpent2 (Serial) vs. WARP\n 4e6 histories, 2MeV point source at origin of 84x84x84cm water block')
pylab.legend(p1,['Serpent 2.1.15 - 18.20 minutes','WARP              -  5.96 minutes'],loc=2)
pylab.ylim([0,.25])
pylab.xlim([1e-11,20])
pylab.grid(True)
pylab.show()
