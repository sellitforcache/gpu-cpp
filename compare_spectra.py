import pylab
import sys
import os

def get_serpent_spec():


if   case=='water':
	tally      = numpy.loadtext('water.tally')
	tallybins  = numpy.loadtext('water.tallybins')
	run ../serpent-benchmark/nonfiss_mono2_det0.m
elif case== 'carbon':
	tally      = numpy.loadtext('carbon.tally')
	tallybins  = numpy.loadtext('carbon.tallybins')
	run ../serpent-benchmark/carbon_mono2_det0.m
elif case== 'lithium':
	tally      = numpy.loadtext('lithium.tally')
	tallybins  = numpy.loadtext('lithium.tallybins')
	#mcnp_tally = numpy.loadtext('../mcnp-benchmark/lithium.tally')
	run ../serpent-benchmark/lithium_mono2_det0.m
elif case== 'hydrogen':
	tally      = numpy.loadtext('hydrogen.tally')
	tallybins  = numpy.loadtext('hydrogen.tallybins')
	run ../serpent-benchmark/hydrogen_mono2_det0.m
elif case== 'hydrogen2':
	tally      = numpy.loadtext('hydrogen2.tally')
	tallybins  = numpy.loadtext('hydrogen2.tallybins')
	run ../serpent-benchmark/hydrogen2_mono2_det0.m

plot_spectra_serpent(DETfluxlogE,DETfluxlog)
hold on
plot_spectra_warp(tally,tallybins,'r')
#plot_spectra_mcnp(mcnp_tally(:,2),mcnp_tally(:,1),'k')

set(gca,'FontSize',14)
title({'Serpent2 (Serial) vs. WARP','4e6 histories, 2MeV point source at origin of 84x84x84cm water block'})
xlabel('Energy (MeV)')
ylabel('Normalized Flux/Lethary')
legend('Serpent2, 18.2 minutes','WARP, 5.76 minutes','MCNP6.1','Location','NW')
grid on
a=ylim();
ylim(a*.5)
xlim([2e-11,25])