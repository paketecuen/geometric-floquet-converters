function nuga_bench_m(libdir, nmax, trials)
% Ciclo de convertidor (3 flujos + 2 saltaciones) con las clases Witt de la version MATLAB de
% NuGA frente a la ruta matricial nativa de MATLAB. Mismas pruebas que el programa en C#.
addpath(libdir); rng(20261119, 'twister');
fprintf('MATLAB %s\n', version);

%% ---------------- BENCH ----------------
fprintf('\n== BENCH ==\n n  refl  err(M_GA,M)  err(bloque V)  signo ok  t_matriz[us] t_compon[us] t_isom[us] t_rotor[us] t_vecGA[us] t_vecGAcomp[us] t_vecM[us]\n');
for pase = 1:2
    for n = 2:nmax
        sp = WittQuadraticSpace.witt(n, n);
        eM = 0; eV = 0; ok = 0; refl = 0; tMat = 0; tComp = 0; tIso = 0; tRot = 0; tVG = 0; tVGc = 0; tVM = 0;
        for t = 1:trials
            c = ciclo(n, [-1.7, 0.6]);
            reps = 2000; tic; for r = 1:reps, M = rutaMatriz(c); end; tMat = tMat + toc/reps*1e6;
            tic; for r = 1:3, wPhi = rotores(sp, c); end; tRot = tRot + toc/(3*numel(wPhi))*1e6;
            wS = saltaciones(sp, c);
            tic; for r = 1:5, W = componer(c, wPhi, wS); end; tComp = tComp + toc/5*1e6;
            tic; for r = 1:20, R = W.getIsometryMatrix(); end; tIso = tIso + toc/20*1e6;
            Mga = R(1:2:end, 1:2:end); Vga = R(2:2:end, 2:2:end); MinvT = inv(M).';
            eM = max(eM, norm(Mga - M, 'fro')/norm(M, 'fro')); eV = max(eV, norm(Vga - MinvT, 'fro')/norm(MinvT, 'fro'));
            ok = ok + (signoEspinorial(sp, W) == sign(det(M))); refl = max(refl, W.Count);
            x = 2*rand(n, 1) - 1; xe = zeros(2*n, 1); xe(1:2:end) = x;
            tic; for r = 1:500, y = W.mapVector(xe); end; tVG = tVG + toc/500*1e6; %#ok<NASGU>
            try
                W.UseCompensatedMapping = true;
                tic; for r = 1:500, y = W.mapVector(xe); end; tVGc = tVGc + toc/500*1e6; %#ok<NASGU>
                W.UseCompensatedMapping = false;
            catch
                tVGc = NaN;
            end
            tic; for r = 1:20000, y = M*x; end; tVM = tVM + toc/20000*1e6; %#ok<NASGU>
        end
        if pase == 2
            fprintf('%2d %5d %12.1e %13.1e %6d/%-3d %11.2f %12.1f %10.1f %11.1f %11.2f %14.2f %11.3f\n', n, refl, eM, eV, ok, trials, ...
                tMat/trials, tComp/trials, tIso/trials, tRot/trials, tVG/trials, tVGc/trials, tVM/trials);
        end
    end
end

%% ---------------- STRESS ----------------
n = 4; sp = WittQuadraticSpace.witt(n, n); nt = 16;
fprintf('\n== STRESS n=4, tres eventos, el segundo degenera; %d ciclos ==\n limite      valor      err mediana   err peor   signo ok   excepciones (primera)\n', nt);
casos = {};
for e = [-2 -4 -6 -8 -10 -12], casos(end+1, :) = {'singular', -1 - 10^e, 10^e}; end %#ok<AGROW>
for e = [2 4 6 8 10],          casos(end+1, :) = {'rasante',  -1 - 10^e, 10^e}; end %#ok<AGROW>
for e = [-2 -4 -6 -8 -10 -12], casos(end+1, :) = {'parabol+',  10^e,     10^e}; end %#ok<AGROW>
for modo = 0:1
    fprintf(' -- mapeo compensado = %d --\n', modo);
    for k = 1:size(casos, 1)
        errs = []; ok = 0; nex = 0; primera = '';
        for t = 1:nt
            c = ciclo(n, [0.6, casos{k, 2}, -0.5]); M = rutaMatriz(c);
            try
                W = componer(c, rotores(sp, c), saltaciones(sp, c));
                if modo == 1, W.UseCompensatedMapping = true; end
                R = W.getIsometryMatrix(); errs(end+1) = norm(R(1:2:end, 1:2:end) - M, 'fro')/norm(M, 'fro'); %#ok<AGROW>
                ok = ok + (signoEspinorial(sp, W) == sign(det(M)));
            catch ex
                nex = nex + 1; if isempty(primera), primera = ex.identifier; end
            end
        end
        if isempty(errs), med = NaN; peor = NaN; else, med = median(errs); peor = max(errs); end
        fprintf(' %-9s %9.1e %12.1e %11.1e %5d/%-3d   %d %s\n', casos{k, 1}, casos{k, 3}, med, peor, ok, nt, nex, primera);
    end
end

%% ---------------- FIRMAS ----------------
fprintf('\n== FIRMAS: la misma rotacion en euclidea(n) y en Witt(n,n) ==\n n | eucl: refl  err   t_vec[us] | Witt: refl  err   t_vec[us] | matriz t_vec[us]\n');
for pase = 1:2
    for n = 2:nmax
        eu = EuclideanQuadraticSpace.euclidean(n); wi = WittQuadraticSpace.witt(n, n);
        eE = 0; eW = 0; tE = 0; tW = 0; tM = 0; rE = 0; rW = 0;
        for t = 1:trials
            G = 2*rand(n) - 1; C = (G - G.')/2; Q = expm(C);
            WE = eu.similitudeFromBivectorCoefficientsMatrix(C, 1.0);
            L = zeros(2*n); L(1:2:end, 1:2:end) = C; L(2:2:end, 2:2:end) = C;
            WW = wi.similitudeFromBivectorIsometryGeneratorMatrix(L, 1.0);
            rE = max(rE, WE.Count); rW = max(rW, WW.Count);
            x = 2*rand(n, 1) - 1; xe = zeros(2*n, 1); xe(1:2:end) = x; yM = Q*x;
            yE = WE.mapVector(x); yW = WW.mapVector(xe);
            eE = max(eE, norm(yE - yM)/norm(yM)); eW = max(eW, norm(yW(1:2:end) - yM)/norm(yM));
            tic; for r = 1:500, y = WE.mapVector(x); end; tE = tE + toc/500*1e6; %#ok<NASGU>
            tic; for r = 1:500, y = WW.mapVector(xe); end; tW = tW + toc/500*1e6; %#ok<NASGU>
            tic; for r = 1:20000, y = Q*x; end; tM = tM + toc/20000*1e6; %#ok<NASGU>
        end
        if pase == 2
            fprintf('%2d | %10d %9.1e %9.2f | %10d %9.1e %9.2f | %12.3f\n', n, rE, eE, tE/trials, rW, eW, tW/trials, tM/trials);
        end
    end
end
end

% ======================================================================
function c = ciclo(n, sigma)
K = numel(sigma); c.n = n; c.K = K; c.sigma = sigma; c.A = cell(1, K+1); c.Phi = cell(1, K+1); c.U = zeros(n, K); c.Nv = zeros(n, K);
for i = 1:K+1
    c.A{i} = (2*rand(n) - 1)*0.6/sqrt(n)*(0.1 + 0.9*rand); c.Phi{i} = expm(c.A{i});
end
for k = 1:K
    d = 0;
    while abs(d) < 0.05, u = 2*rand(n, 1) - 1; nv = 2*rand(n, 1) - 1; d = nv.'*u; end
    c.U(:, k) = u*(sigma(k)/d); c.Nv(:, k) = nv;
end
end
function M = rutaMatriz(c)
M = c.Phi{1};
for k = 1:c.K, M = M + c.U(:, k)*(c.Nv(:, k).'*M); M = c.Phi{k+1}*M; end
end
function w = rotores(sp, c)
n = c.n; w = cell(1, c.K+1);
for i = 1:c.K+1
    L = zeros(2*n); L(1:2:end, 1:2:end) = c.A{i}; L(2:2:end, 2:2:end) = -c.A{i}.';
    w{i} = sp.similitudeFromBivectorIsometryGeneratorMatrix(L, 1.0);
end
end
function w = saltaciones(sp, c)
n = c.n; w = cell(1, c.K);
for k = 1:c.K
    s = c.sigma(k); p = zeros(2*n, 1); q = zeros(2*n, 1);
    p(1:2:end) = c.U(:, k);   p(2:2:end) = c.Nv(:, k);              % <p,p> = 2 sigma
    q(1:2:end) = c.U(:, k)/s; q(2:2:end) = (1 + 1/s)*c.Nv(:, k);    % <q,q> = 2 (1+sigma)/sigma
    w{k} = sp.similitude([p, q], 1.0);
end
end
function W = componer(c, wPhi, wS)
W = wPhi{1};
for k = 1:c.K, W = wPhi{k+1}.compose(wS{k}.compose(W)); end
end
function s = signoEspinorial(sp, W)
s = 1; for i = 1:W.Count, s = s*sign(sp.normSquared(W.getReflectionVector(i))); end
end
