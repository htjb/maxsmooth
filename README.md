# msf-opt/maxsmooth

Code needs some tidying I am sure but if you git clone this and run test.py it should do some nice fits/residuals/derivatives graphs and save some run time data. You can play around with the settings (I realise my descriptions will also need tidying!). If the code throws up an error it will give advice on what to do to make it work hopefully. Some settings combinations will also give warnings.

You can allow inflection points to occur by changing some of the settings. Currently you have to state which derivative you would like to allow an inflection point in.

There is also a range of basis functions to try. 'normalised_polynomial' works best.

## **Tasks**
- [ ] Tidy up variable/class/function names.
- [ ] Simplify  settings descriptions.
- [ ] Maybe edit the way the run time data is stored for easier access.
- [ ] Add in potential for user to input their own basis function.
- [ ] Maybe include a 'test all' feature for inflection points. (This will get complicated/time consuming if the user decides to allow inflection points in two of eleven derivatives for a 13th order polynomial for example and isn't really the point of the software).


