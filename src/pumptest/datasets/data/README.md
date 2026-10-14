# Example datasets

The Gridley, Sioux Flats and Texas Hill files are copied unchanged from the documentation of TTim,
<https://github.com/mbakker7/ttim>, `docs/03pumpingtests/data/`.
TTim is released under the MIT licence:

> The MIT License (MIT)
>
> Copyright (c) 2015 mbakker7
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

| File | Test | Original source |
|---|---|---|
| `gridley_well_1.txt`, `_well_3.txt` | Gridley, Illinois, confined | Walton (1962), via the AQTESOLV manual |
| `sioux100.txt`, `200`, `400` | Sioux Flats, South Dakota | AQTESOLV manual (Duffield, 2007) |
| `texas40.txt`, `80`, `160` | Texas Hill, leaky | AQTESOLV manual (Duffield, 2007) |

The AQTESOLV parameter values the tests compare against are those tabulated
in the TTim notebooks, which take them from Yang, X. (2020), *Application and
comparison of different methods for aquifer test analysis using TTim*, MSc
thesis, TU Delft.

## Lauswiesen, Tuebingen (CC BY-SA 4.0)

`lauswiesen_B2.txt.gz` to `lauswiesen_B5.txt.gz` hold four pumping tests at the
Lauswiesen research site, each pumped in one of wells B2 to B5 and logged once a
second in all five wells B1 to B5 (time in seconds, drawdown in metres).

- Data: Leven, C. (2020), *Pumping Test Data from the Hydrogeological Research
  Site "Lauswiesen"*, University of Tuebingen, https://doi.org/10.57754/fdat.hma1z-00r32,
  licensed under Creative Commons Attribution-ShareAlike 4.0
  (https://creativecommons.org/licenses/by-sa/4.0/). These four files keep
  that licence; the rest of pumptest is MIT.
- Converted by `scripts/import_lauswiesen.py` from `data/lauswiesen.cmp` in the
  archived workflow of Mueller, S., Leven, C., Dietrich, P., Attinger, S. and
  Zech, A. (2022), "How to find aquifer statistics utilizing pumping tests? Two
  field studies using welltestpy", *Groundwater* 60(1), 137-144,
  https://doi.org/10.1111/gwat.13121; workflow at
  https://doi.org/10.5281/zenodo.4724118 (MIT). Only the file format changed.
- Published values: the Theis estimates of that workflow
  (`results/01b_estimate/<test>/estimate.txt`, mean ln T in m2/s and ln S),
  fitted by welltestpy to all five wells of each test.
  welltestpy does not fit the raw records: it normalises the rate, low-pass
  filters each record and keeps ten quadratically spaced times per well from
  10 s. `tests/test_validation_welltestpy.py` repeats that preprocessing and
  reproduces the published estimates; fits to the raw records differ from
  them by up to 3.5 percent in transmissivity.
