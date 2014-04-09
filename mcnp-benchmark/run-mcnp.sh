#! /bin/bash

mcnp6 i=godiva       o=godiva.o   mctal=godiva.tally
mcnp6 i=homfuelcrit  o=homfuel.o  mctal=homfuel.tally
mcnp6 i=uh2opincell  o=pincell.o  mctal=pincell.tally
mcnp6 i=uh2oassembly o=assembly.o mctal=assembly.tally

