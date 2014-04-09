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

def get_mcnp_det(filepath):
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
	serpdata   = get_serpent_det('../serpent-benchmark/water_10Mev_det0.m')
elif case=='isowater':
	tally      = numpy.loadtxt('isowater.tally')
	tallybins  = numpy.loadtxt('isowater.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/nonfiss_mono2_det0.m')
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
elif case== 'oxygen16':
	tally      = numpy.loadtxt('o16.tally')
	tallybins  = numpy.loadtxt('o16.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/oxygen16_mono2_det0.m')
elif case== 'aluminum':
	tally      = numpy.loadtxt('aluminum.tally')
	tallybins  = numpy.loadtxt('aluminum.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/aluminum_mono2_det0.m')
elif case== 'u235':
	tally      = numpy.loadtxt('u235.tally')
	tallybins  = numpy.loadtxt('u235.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/u235_mono2_det0.m')
elif case== 'isou235':
	tally      = numpy.loadtxt('isou235.tally')
	tallybins  = numpy.loadtxt('isou235.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/u235_mono2_iso_det0.m')
elif case== '1ku235':
	tally      = numpy.loadtxt('1ku235.tally')
	tallybins  = numpy.loadtxt('1ku235.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/u235_mono1k_det0.m')
elif case== '1evu235':
	tally      = numpy.loadtxt('1evu235.tally')
	tallybins  = numpy.loadtxt('1evu235.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/u235_mono1ev_det0.m')
elif case== '1kpu239':
	tally      = numpy.loadtxt('1kpu239.tally')
	tallybins  = numpy.loadtxt('1kpu239.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/pu239_mono1k_det0.m')
elif case== 'pb':
	tally      = numpy.loadtxt('lead.tally')
	tallybins  = numpy.loadtxt('lead.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/pb208_15M_det0.m')
elif case== 'u235-crit':
	tally      = numpy.loadtxt('u235_crit.tally')
	tallybins  = numpy.loadtxt('u235_crit.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/u235_crit_det0.m')
elif case== 'u235238-crit':
	tally      = numpy.loadtxt('u235238_crit.tally')
	tallybins  = numpy.loadtxt('u235238_crit.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/u235238_crit_det0.m')
elif case== 'u238-crit':
	tally      = numpy.loadtxt('u238_crit.tally')
	tallybins  = numpy.loadtxt('u238_crit.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/u238_crit_det0.m')
elif case== 'u235-crit-nocont':
	tally      = numpy.loadtxt('u235_crit_nocont.tally')
	tallybins  = numpy.loadtxt('u235_crit_nocont.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/u235_crit_det0.m')
elif case== 'homfuel':
	tally      = numpy.loadtxt('homfuel.tally')
	tallybins  = numpy.loadtxt('homfuel.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/homfuel_crit_det0.m')
	title = 'Serpent2 (Serial) vs. WARP 6e6 histories (2e6 discarded)\n Flux in homogenized block of UO2 and water'	
elif case== 'uh2o-pincell':
	tally      = numpy.loadtxt('uh2o-pincell.tally')
	tallybins  = numpy.loadtxt('uh2o-pincell.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/uh2o-pincell_det0.m')
	title='Serpent2 (Serial) vs. WARP 6e6 histories (2e6 discarded)\n Flux in the water of surrpunding a single UO2 pin'
elif case== 'godiva':
	tally      = numpy.loadtxt('godiva.tally')
	tallybins  = numpy.loadtxt('godiva.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/godiva_det0.m')
	title = 'Serpent2 (Serial) vs. WARP 6e6 histories (2e6 discarded)\n Flux in a bare Pu-239 sphere (Godiva)'
elif case== 'assembly':
	tally      = numpy.loadtxt('uh2o-assembly.tally')
	tallybins  = numpy.loadtxt('uh2o-assembly.tallybins')
	serpdata   = get_serpent_det('../serpent-benchmark/uh2o-assembly_det0.m')
	title = 'Serpent2 (Serial) vs. WARP 6e6 histories (2e6 discarded)\n Flux in the water of a hexagonal array of UO2 pins'



widths=numpy.diff(tallybins);
avg=(tallybins[:-1]+tallybins[1:])/2;
#newflux = numpy.multiply(tally[:-1,1],tally[:-1,0])
#newflux = numpy.divide(newflux,numpy.add(tally[:-1,1],1.0))
newflux=numpy.array(tally[:,0])
warp_err = numpy.array(tally[:,1])
newflux=numpy.divide(newflux,widths)
newflux=numpy.multiply(newflux,avg)
#newflux=numpy.divide(newflux,40e5)
#newflux=numpy.divide(newflux,1)#numpy.max(newflux))

serpE=numpy.array(serpdata['DETfluxlogE'][:,2])
serpErr=numpy.array(serpdata['DETfluxlog'][:,11])
serpF=numpy.array(serpdata['DETfluxlog'][:,10])
#serpF=numpy.divide(serpdata['DETfluxlog'][:,10],1)#numpy.max(serpdata['DETfluxlog'][:,10]))
serpE = numpy.squeeze(numpy.asarray(serpE))
serpErr = numpy.squeeze(numpy.asarray(serpErr))
serpF = numpy.squeeze(numpy.asarray(serpF))

fig = pylab.figure(figsize=(10,6))
ax = fig.add_subplot(1,1,1)
ax.semilogx(serpE,serpF,'b',linestyle='steps-mid',label='Serpent 2.1.15')
ax.semilogx(avg,newflux,'r',linestyle='steps-mid',label='WARP')
ax.set_xlabel('Energy (MeV)')
ax.set_ylabel('Normalized Flux/Lethary')
ax.set_title(title)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles,labels,loc=2)
ax.set_xlim([1e-11,20])
ax.grid(True)
if len(sys.argv)==2:
	pylab.show()
else:
	print 'spec.eps'
	fig.savefig('spec.eps')

fig = pylab.figure(figsize=(10,6))
ax = fig.add_subplot(1,1,1)
ax.semilogx(serpE,serpErr,'b',linestyle='steps-mid',label='Serpent Rel. Err.')
ax.semilogx(avg,warp_err,'r',linestyle='steps-mid',label='WARP Rel. Err.')
ax.semilogx(serpE,numpy.divide(serpF-newflux,serpF),'g',linestyle='steps-mid',label='Flux Relative Error')
ax.set_xlabel('Energy (MeV)')
ax.set_title(title)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles,labels,loc=2)
#pylab.ylim([0,.25])
ax.set_xlim([1e-11,20])
ax.set_ylim([-1e-1,1e-1])
ax.grid(True)
if len(sys.argv)==2:
	pylab.show()
else:
	print 'spec_err.eps'
	fig.savefig('spec_err.eps')
