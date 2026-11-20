// Banco de prueba: el ciclo de un convertidor conmutado (flujos + saltaciones) construido con las
// clases Witt de NuGA (versor = lista de normales de reflexion en R^{2n}, signatura (n,n)) frente a
// la ruta matricial n x n. Mismo runtime, mismos datos.
//   modos:  bench <nmax> <trials>   |   stress <n> <trials>   |   eval <nmax> <trials>
using System.Diagnostics;
using System.Globalization;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using GeometricAlgebraFulcrumLib.Algebra.LinearAlgebra.Numerical.QuadraticSpaces.Spaces;
using GeometricAlgebraFulcrumLib.Algebra.LinearAlgebra.Numerical.QuadraticSpaces.Similitudes;
using GeometricAlgebraFulcrumLib.Algebra.LinearAlgebra.Numerical.QuadraticSpaces.Validations;

static class Bench
{
    static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

    static Matrix<double> Expm(Matrix<double> a)
    {
        int n = a.RowCount; double nrm = a.InfinityNorm();
        int s = nrm > 0.5 ? (int)Math.Ceiling(Math.Log2(nrm / 0.5)) : 0;
        var x = a / Math.Pow(2, s);
        Matrix<double> term = DenseMatrix.CreateIdentity(n), sum = DenseMatrix.CreateIdentity(n);
        for (int k = 1; k < 30; k++) { term = term * x / k; sum += term; if (term.InfinityNorm() < 1e-19) break; }
        for (int i = 0; i < s; i++) sum = sum * sum;
        return sum;
    }
    // base de Witt intercalada de NuGA: indices pares = vectores (w_i), impares = covectores (v_i)
    static Matrix<double> Interleave(Matrix<double> onW, Matrix<double> onV)
    {
        int n = onW.RowCount; var r = new DenseMatrix(2 * n, 2 * n);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { r[2 * i, 2 * j] = onW[i, j]; r[2 * i + 1, 2 * j + 1] = onV[i, j]; }
        return r;
    }
    static Vector<double> Embed(Vector<double> u, Vector<double> nv, double cu, double cn)
    {
        int n = u.Count; var p = new DenseVector(2 * n);
        for (int i = 0; i < n; i++) { p[2 * i] = cu * u[i]; p[2 * i + 1] = cn * nv[i]; }
        return p;
    }

    sealed class Cycle
    {
        public int N, K; public Matrix<double>[] A = null!, Phi = null!; public Vector<double>[] U = null!, Nv = null!; public double[] Sigma = null!;
    }
    static Cycle RandomCycle(Random rng, int n, double[] sigma)
    {
        int K = sigma.Length; var c = new Cycle { N = n, K = K, Sigma = sigma, A = new Matrix<double>[K + 1], Phi = new Matrix<double>[K + 1], U = new Vector<double>[K], Nv = new Vector<double>[K] };
        for (int i = 0; i <= K; i++)
        {
            var a = DenseMatrix.Build.Dense(n, n, (r, col) => (rng.NextDouble() * 2 - 1) * 0.6 / Math.Sqrt(n));
            c.A[i] = a * (0.1 + 0.9 * rng.NextDouble()); c.Phi[i] = Expm(c.A[i]);
        }
        for (int k = 0; k < K; k++)
        {
            Vector<double> u, nv; double d;
            do { u = DenseVector.Build.Dense(n, _ => rng.NextDouble() * 2 - 1); nv = DenseVector.Build.Dense(n, _ => rng.NextDouble() * 2 - 1); d = nv * u; } while (Math.Abs(d) < 0.05);
            c.U[k] = u * (sigma[k] / d); c.Nv[k] = nv;
        }
        return c;
    }
    static Matrix<double> MatrixRoute(Cycle c)
    {
        Matrix<double> M = c.Phi[0];
        for (int k = 0; k < c.K; k++) { var row = c.Nv[k] * M; M = M + c.U[k].OuterProduct(row); M = c.Phi[k + 1] * M; }
        return M;
    }
    static WittSimilitude[] Rotors(WittQuadraticSpace space, Cycle c)
        => c.A.Select(a => space.SimilitudeFromBivectorIsometryGeneratorMatrix(Interleave(a, -a.Transpose()))).ToArray();
    static WittSimilitude[] Saltations(WittQuadraticSpace space, Cycle c)
    {
        var ws = new WittSimilitude[c.K];
        for (int k = 0; k < c.K; k++)
        {
            var p = Embed(c.U[k], c.Nv[k], 1.0, 1.0);                                   // <p,p> = 2 sigma
            var q = Embed(c.U[k], c.Nv[k], 1.0 / c.Sigma[k], 1.0 + 1.0 / c.Sigma[k]);    // <q,q> = 2 (1+sigma)/sigma
            ws[k] = space.Similitude(DenseMatrix.Build.DenseOfColumnVectors(p, q), 1.0);
        }
        return ws;
    }
    static WittSimilitude Compose(Cycle c, WittSimilitude[] wPhi, WittSimilitude[] wS)
    {
        var W = wPhi[0];
        for (int k = 0; k < c.K; k++) W = wPhi[k + 1].Compose(wS[k].Compose(W));
        return W;
    }
    static (double eM, double eV, double mix) Compare(Matrix<double> R, Matrix<double> M)
    {
        int n = M.RowCount; var Mga = new DenseMatrix(n, n); var Vga = new DenseMatrix(n, n); double mix = 0;
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
        { Mga[i, j] = R[2 * i, 2 * j]; Vga[i, j] = R[2 * i + 1, 2 * j + 1]; mix = Math.Max(mix, Math.Max(Math.Abs(R[2 * i, 2 * j + 1]), Math.Abs(R[2 * i + 1, 2 * j]))); }
        var MinvT = M.Inverse().Transpose();
        return ((Mga - M).FrobeniusNorm() / M.FrobeniusNorm(), (Vga - MinvT).FrobeniusNorm() / MinvT.FrobeniusNorm(), mix);
    }
    static int SpinorSign(WittQuadraticSpace space, WittSimilitude W)
    { int s = 1; for (int i = 0; i < W.Count; i++) s *= Math.Sign(space.NormSquared(W.GetReflectionVector(i))); return s; }
    static double Median(List<double> v) { var s = v.OrderBy(x => x).ToArray(); return s.Length == 0 ? double.NaN : s[s.Length / 2]; }

    public static void Main(string[] args)
    {
        CultureInfo.DefaultThreadCurrentCulture = Inv; CultureInfo.CurrentCulture = Inv;
        string mode = args.Length > 0 ? args[0] : "bench";
        int a1 = args.Length > 1 ? int.Parse(args[1]) : 6, trials = args.Length > 2 ? int.Parse(args[2]) : 10;
        var rng = new Random(20261119);
        Console.WriteLine($"runtime {Environment.Version}  modo {mode}");
        if (mode == "eval")
        {
            PseudoEuclideanAccuracyEvaluator.RunFullBenchmarks(Enumerable.Range(2, Math.Max(1, a1 - 1)).Select(k => (k, k)).ToArray(), trials);
            return;
        }
        if (mode == "firmas")
        {
            // Un mismo factor conservativo (rotacion e^C, C antisimetrica) en dos signaturas:
            // euclidea de dimension n (adaptada) y lift universal en Witt(n,n) de dimension 2n.
            Console.WriteLine(" n | euclidea(n): refl  err      t_vec[us] | Witt(n,n): refl  err      t_vec[us] | matriz t_vec[us] | Witt/eucl  eucl/matriz");
            for (int pase = 0; pase < 2; pase++)
            for (int n = 2; n <= a1; n++)
            {
                var eu = EuclideanQuadraticSpace.Euclidean(n); var wi = WittQuadraticSpace.Witt(n, n);
                double eE = 0, eW = 0, tE = 0, tW = 0, tM = 0; int rE = 0, rW = 0;
                for (int t = 0; t < trials; t++)
                {
                    var g = DenseMatrix.Build.Dense(n, n, (r, col) => rng.NextDouble() * 2 - 1); Matrix<double> C = (g - g.Transpose()) * 0.5;
                    var Q = Expm(C);
                    var WE = eu.SimilitudeFromBivectorCoefficientsMatrix(C);
                    var WW = wi.SimilitudeFromBivectorIsometryGeneratorMatrix(Interleave(C, C));   // -C^T = C
                    rE = Math.Max(rE, WE.Count); rW = Math.Max(rW, WW.Count);
                    var x = DenseVector.Build.Dense(n, _ => rng.NextDouble() * 2 - 1); var xe = Embed(x, x, 1.0, 0.0);
                    var yM = Q * x; var yE = WE.MapVector(x); var yW = WW.MapVector(xe);
                    var yWw = DenseVector.Build.Dense(n, i => yW[2 * i]);
                    eE = Math.Max(eE, (yE - yM).L2Norm() / yM.L2Norm()); eW = Math.Max(eW, (yWw - yM).L2Norm() / yM.L2Norm());
                    var sw = Stopwatch.StartNew(); for (int r = 0; r < 5000; r++) { var y = WE.MapVector(x); } tE += sw.Elapsed.TotalMilliseconds * 1000 / 5000;
                    sw.Restart(); for (int r = 0; r < 5000; r++) { var y = WW.MapVector(xe); } tW += sw.Elapsed.TotalMilliseconds * 1000 / 5000;
                    sw.Restart(); for (int r = 0; r < 20000; r++) { var y = Q * x; } tM += sw.Elapsed.TotalMilliseconds * 1000 / 20000;
                }
                if (pase == 1) Console.WriteLine(string.Format(Inv, "{0,2} | {1,17} {2,9:E1} {3,9:F3} | {4,15} {5,9:E1} {6,9:F3} | {7,13:F3}    | {8,8:F1} {9,11:F1}",
                    n, rE, eE, tE / trials, rW, eW, tW / trials, tM / trials, tW / tE, tE / tM));
            }
            return;
        }
        if (mode == "stress")
        {
            int n = a1; var space = WittQuadraticSpace.Witt(n, n);
            Console.WriteLine($"n={n}, tres eventos, el segundo degenera; mediana sobre {trials} ciclos");
            Console.WriteLine(" limite     |1+sigma| (o |sigma| en parabol)   err M_GA vs M (mediana)   err peor   signo ok");
            var casos = new List<(string, double)>();
            foreach (var e in new[] { -2, -4, -6, -8, -10, -12 }) casos.Add(("singular", -1.0 - Math.Pow(10, e)));
            foreach (var e in new[] { 2, 4, 6, 8, 10 }) casos.Add(("rasante", -1.0 - Math.Pow(10, e)));
            foreach (var e in new[] { -2, -4, -6, -8, -10, -12 }) casos.Add(("parabol+", Math.Pow(10, e)));      // sigma -> 0+: evento PWM
            foreach (var e in new[] { -2, -6, -10 }) casos.Add(("parabol-", -Math.Pow(10, e)));                  // sigma -> 0-
            foreach (var (nombre, sig) in casos)
            {
                var errs = new List<double>(); int ok = 0, fallos = 0;
                for (int t = 0; t < trials; t++)
                {
                    var c = RandomCycle(rng, n, new[] { 0.6, sig, -0.5 });
                    var M = MatrixRoute(c);
                    try
                    {
                        var W = Compose(c, Rotors(space, c), Saltations(space, c));
                        var (eM, _, _) = Compare(W.GetIsometryMatrix(), M); errs.Add(eM);
                        if (SpinorSign(space, W) == Math.Sign(M.Determinant())) ok++;
                    }
                    catch (Exception ex) { fallos++; if (fallos == 1) Console.WriteLine($"   (excepcion: {ex.GetType().Name}: {ex.Message.Split('\n')[0]})"); }
                }
                Console.WriteLine(string.Format(Inv, " {0,-9} {1,10:E1} {2,18:E1} {3,18:E1} {4,10}/{5} {6}", nombre, nombre.StartsWith("parabol") ? Math.Abs(sig) : Math.Abs(1 + sig), Median(errs), errs.Count > 0 ? errs.Max() : double.NaN, ok, trials, fallos > 0 ? $"({fallos} excepciones)" : ""));
            }
            return;
        }
        Console.WriteLine(" n  dim  refl propio  err(M_GA,M)  err(bloque V)  mezcla    signo ok   t_matriz[us]  t_compon[us]  t_isometria[us]  t_rotor[us]  t_vec_GA[us]  t_vec_M[us]");
        for (int pase = 0; pase < 2; pase++)          // pase 0: calentamiento del JIT, no se imprime
        for (int n = 2; n <= a1; n++)
        {
            var space = WittQuadraticSpace.Witt(n, n);
            double eM = 0, eV = 0, eMix = 0; int okSign = 0, refl = 0; bool proper = true;
            double tMat = 0, tComp = 0, tIso = 0, tRot = 0, tVecGa = 0, tVecM = 0;
            for (int t = 0; t < trials; t++)
            {
                var c = RandomCycle(rng, n, new[] { -1.7, 0.6 });
                Matrix<double> M = MatrixRoute(c);
                var sw = Stopwatch.StartNew(); int reps = 5000;
                for (int r = 0; r < reps; r++) M = MatrixRoute(c);
                tMat += sw.Elapsed.TotalMilliseconds * 1000 / reps;
                var wPhi = Rotors(space, c);
                sw.Restart(); for (int r = 0; r < 20; r++) wPhi = Rotors(space, c);
                tRot += sw.Elapsed.TotalMilliseconds * 1000 / (20 * (c.K + 1));
                var wS = Saltations(space, c);
                var W = Compose(c, wPhi, wS);
                sw.Restart(); for (int r = 0; r < 200; r++) W = Compose(c, wPhi, wS);
                tComp += sw.Elapsed.TotalMilliseconds * 1000 / 200;
                Matrix<double> R = W.GetIsometryMatrix();
                sw.Restart(); for (int r = 0; r < 200; r++) R = W.GetIsometryMatrix();
                tIso += sw.Elapsed.TotalMilliseconds * 1000 / 200;
                var (e1, e2, e3) = Compare(R, M); eM = Math.Max(eM, e1); eV = Math.Max(eV, e2); eMix = Math.Max(eMix, e3);
                if (SpinorSign(space, W) == Math.Sign(M.Determinant())) okSign++;
                refl = Math.Max(refl, W.Count); proper &= W.IsProperRotation;
                var x = DenseVector.Build.Dense(n, _ => rng.NextDouble() * 2 - 1); var xe = Embed(x, x, 1.0, 0.0);
                sw.Restart(); for (int r = 0; r < 2000; r++) { var y = W.MapVector(xe); } tVecGa += sw.Elapsed.TotalMilliseconds * 1000 / 2000;
                sw.Restart(); for (int r = 0; r < 20000; r++) { var y = M * x; } tVecM += sw.Elapsed.TotalMilliseconds * 1000 / 20000;
            }
            if (pase == 1) Console.WriteLine(string.Format(Inv, "{0,2} {1,4} {2,5} {3,6} {4,12:E1} {5,14:E1} {6,9:E1} {7,6}/{8,-4} {9,12:F2} {10,13:F2} {11,16:F1} {12,12:F1} {13,13:F2} {14,12:F3}",
                n, 2 * n, refl, proper, eM, eV, eMix, okSign, trials, tMat / trials, tComp / trials, tIso / trials, tRot / trials, tVecGa / trials, tVecM / trials));
        }
    }
}
