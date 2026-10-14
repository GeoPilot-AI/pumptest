# Example datasets

The `.txt` and `.dat` files below (all except `hoogezand.txt` and `well1_steps.txt`) are copied unchanged from the documentation of TTim,
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
| `oudekorendijk_h30.dat`, `_h90.dat` | Oude Korendijk, confined | Kruseman and de Ridder (1970); time in minutes |
| `gridley_well_1.txt`, `_well_3.txt` | Gridley, Illinois, confined | Walton (1962), via the AQTESOLV manual |
| `sioux100.txt`, `200`, `400` | Sioux Flats, South Dakota | AQTESOLV manual (Duffield, 2007) |
| `texas40.txt`, `80`, `160` | Texas Hill, leaky | AQTESOLV manual (Duffield, 2007) |
| `dalem_p30.txt` to `p120` | Dalem, leaky | Kruseman and de Ridder (1970) |
| `venne_deep.txt` | Vennebulten, unconfined, deep piezometer at 90 m | Kruseman and de Ridder (1970); time in minutes |

The AQTESOLV, MLU and Kruseman and de Ridder parameter values the tests compare
against are those tabulated in the TTim notebooks, which take them from
Yang, X. (2020), *Application and comparison of different methods for aquifer
test analysis using TTim*, MSc thesis, TU Delft.

## Worked examples from Kruseman and de Ridder

Transcribed by hand from Kruseman, G.P. and de Ridder, N.A. (1994), *Analysis
and evaluation of pumping test data*, 2nd edition, ILRI Publication 47,
Wageningen. The page numbers are those printed in the book.

| File | Example | Table, page | Published result |
|---|---|---|---|
| `well1_steps.txt` | Step-drawdown "Well 1" (Clark 1977) | Table 14.2, p. 205 | B = 3.26e-3 d/m2, C = 1.45e-7 d2/m5 (Hantush-Bierschenk, graphical) |
| `hoogezand.txt` | Single-well test "Hoogezand" (Mulder 1983) | Table 15.1, p. 224 | KD = 1800 m2/d (Jacob, t > 41 s); S = 4e-4, c = 1000 d after Mulder |

The Vennebulten unconfined test (Table 5.1, p. 106) is the same data as
`venne_deep.txt`; the published Neuman analysis (pp. 104 to 106) gives
KhD = 1447 m2/d (type A) and 1616 m2/d (type B), SA = 5.2e-4, Sy = 4.9e-3 and
beta = 0.01. The book itself calls that Sy unreasonably low.

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
