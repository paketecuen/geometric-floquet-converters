# Geometric Algebra Lifts and Scalar Floquet Criteria for Switched Power Converters

Manuscript, figure scripts and numerical verification for

> F. G. Montoya and A. Alcayde, *Geometric Algebra Lifts and Scalar Floquet
> Criteria for Switched Power Converters.* Submitted to IEEE Transactions on
> Circuits and Systems I (first revision, November 2026).

This work has been submitted to the IEEE for possible publication. Copyright
may be transferred without notice, after which this version may no longer be
accessible.

University of Almería, Department of Engineering.

## Layout

```
TCAS-I/
  main.tex, main.pdf             manuscript (IEEEtran, compiles with latexmk/pdflatex)
  supplement.tex, supplement.pdf supplementary material (conditioning table, ensemble)
  figs/
    estilo_ieee.py               shared style (pgf + usetex, IEEE column width) and the
                                 text-overlap detector every figure script asserts on
    fig_tutorial.py              Fig. 1  circuit, one switching period, switching function
    fig0_geometria.py            Fig. 2  saltation geometry and rapidity sheets
    fig1_bifurcation.py          Fig. 3  subharmonic boundary, amplitude fit
    fig2_parity.py               Fig. 4  parity under broken symmetry
    fig5_par_flip.py             Fig. 5  even case of the parity rule: beat frequency against
                                 mismatch (R1); also Fig. S2 of the supplement
    fig3_rapidity.py             Fig. 6  CCM / DCM / duty-saturation sweep
    fig4_retardo_esquina.py      Fig. 7  digital delay (exp5); the simultaneous-events panel
                                 (exp6) is Fig. S1 of the supplement since R1
    verifica_lift.py             Prop. 1 in lift form: Q = 1 + u n, generic u, n
    verifica_grados.py           Prop. 3: grades of the lift give det(I+M), det(I-M),
                                 det(I+zM); flip boundary of the buck without eigenvalues
    verifica_apendice.py         the exponential forms of Appendix A
    verifica_ejemplo_versor.py   the running example of Sec. III
    verifica_especies.py         species traversal of Table III, DCM counterexample
    verifica_rapidez_rampa.py    rapidity with a compensation ramp
    verifica_numeros_exactos.py  exact roots behind every published number
    verifica_ejemplo_completo.py complete two-state example and normal-form parameters (R1)
    verifica_metrica_energia.py  interval flows as near-rotations in the energy metric (R1)
  validacion/                    falsification campaign (exp1..exp10, comun.py), results in
                                 resultados/*.json, with its own README
  validacion/exp11_par_flip/     even case of the parity rule (Sec. V-D, R1): closed form,
                                 event-driven time-domain simulator, published boost data
  buck_realista.py, interleaved_realista.py, dcm_realista.py, blindaje_paridad.py
                                 converter models the figures derive from
R1/                              first revision: clean manuscript, manuscript with the
                                 changes marked, and supplementary material
requirements.txt                 pinned Python dependencies
```

## Reproducing

Python 3.9+ with the pinned packages of `requirements.txt`, plus a TeX
installation (the figure scripts render their text with `pdflatex`, so the
typography matches the manuscript). Create the virtual environment **outside
any synced folder** (Google Drive, Dropbox): a venv inside a synced folder
floods the sync client with thousands of files.

```bash
python3 -m venv ~/venvs/floquet_GA
source ~/venvs/floquet_GA/bin/activate
pip install -r requirements.txt
cd TCAS-I/figs
python fig_tutorial.py              # and fig0 ... fig5
python verifica_lift.py             # and the other verifica_*.py
cd ../validacion
python exp1_falsacion_paridad.py    # slow; exp2 ... exp10 take seconds to minutes
cd exp11_par_flip                   # even case of the parity rule (Sec. V-D)
python par_flip_interleaved.py      # closed form and monodromy
python td1_validacion.py            # time-domain simulator against the monodromy
python td2_atractor.py && python td3_umbral.py   # minutes, 12 processes
cd ../.. && latexmk -pdf main.tex && latexmk -pdf supplement.tex
```

Every figure script ends in an assertion on the text-overlap detector, so a
layout regression fails loudly. Every `verifica_*.py` ends in assertions on
the numerical agreement, and every `exp*.py` exits with a nonzero code if it
finds a counterexample.

## Not included

Publisher PDFs of the cited literature are copyrighted and are not
redistributed. The correspondence with the journal (decision letter,
reviewers' comments and the response to them) is confidential and is not
part of the repository.

The programs of `TCAS-I/validacion/nuga/` (Tables S2 and S3 of the
supplement) need the numerical library of A. H. Eid,
[GeometricAlgebraFulcrumLib](https://github.com/ga-explorer/GeometricAlgebraFulcrumLib),
which is not redistributed here. They were run with the version of
19 November 2026, provided by its author ahead of its release in that
repository.
