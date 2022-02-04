Python "wrapper for the ExtractChem equilbrium geochemical model."

Created by A.J. Brown
ansleybrown1337@gmail.com

This set of code allows you to automatically generate batch files from a csv template containing various soil characteristics, and then graph certain outputs after running the batch file in the ExtractChem model. It can be a helpful tool for Monte Carlo analysis for understanding sensitivities of soil samples to ion adsorption, dissolution and precipitation during a change in moisture content, and the resulting impact it has on soil solution electrical conductivity.

The general workflow is this:
1. Run ExtractChem_batch_merge.py to create batch file
2. Open ExtractChem manually (unfortuately it cannot be ran via terminal) and run the batch file    generated in step 1.
3. Run EC_graphs.py for collecting output and displaying results. This script requires the custom    GOF_stats.py module to function properly, and is included in the Code folder.

I haven't had time to make much help documentation on this yet, so I'm happy to help if you reach out to me for specific applications and tutorials.

Cheers,
AJ