function nuga_concat_m(libdir, nmax, trials)
% Igual que nuga_bench_m pero componiendo por CONCATENACION de las normales (como hace la
% version C#), sin pasar por compose(), que multiplica las matrices y vuelve a factorizar.
addpath(libdir); rng(20261119, 'twister');
fprintf('MATLAB %s  (composicion por concatenacion)\n', version);
fprintf('\n== BENCH ==\n n  refl  err(M_GA,M) plano  err compensado  signo ok  t_matriz[us] t_concat[us] t_vecGA[us] t_vecGAcomp[us] t_vecM[us]\n');
for pase = 1:2
    for n = 2:nmax
        sp = WittQuadraticSpace.witt(n, n); eM = 0; eMc = 0; ok = 0; refl = 0; tMat = 0; tC = 0; tVG = 0; tVGc = 0; tVM = 0;
        for t = 1:trials
            c = ciclo(n, [-1.7, 0.6]);
            tic; for r = 1:2000, M = rutaMatriz(c); end; tMat = tMat + toc/2000*1e6;
            wPhi = rotores(sp, c); wS = saltaciones(sp, c);
            tic; for r = 1:20, W = concatenar(sp, c, wPhi, wS); end; tC = tC + toc/20*1e6;
            R = W.getIsometryMatrix(); eM = max(eM, norm(R(1:2:end, 1:2:end) - M, 'fro')/norm(M, 'fro'));
            W.UseCompensatedMapping = true; R = W.getIsometryMatrix(); eMc = max(eMc, norm(R(1:2:end, 1:2:end) - M, 'fro')/norm(M, 'fro'));
            ok = ok + (signoEspinorial(sp, W) == sign(det(M))); refl = max(refl, W.Count);
            x = 2*rand(n, 1) - 1; xe = zeros(2*n, 1); xe(1:2:end) = x;
            tic; for r = 1:300, y = W.mapVector(xe); end; tVGc = tVGc + toc/300*1e6; %#ok<NASGU>
            W.UseCompensatedMapping = false;
            tic; for r = 1:300, y = W.mapVector(xe); end; tVG = tVG + toc/300*1e6; %#ok<NASGU>
            tic; for r = 1:20000, y = M*x; end; tVM = tVM + toc/20000*1e6; %#ok<NASGU>
        end
        if pase == 2
            fprintf('%2d %5d %14.1e %16.1e %6d/%-3d %11.2f %11.1f %11.2f %14.2f %11.3f\n', n, refl, eM, eMc, ok, trials, tMat/trials, tC/trials, tVG/trials, tVGc/trials, tVM/trials);
        end
    end
end
n = 4; sp = WittQuadraticSpace.witt(n, n); nt = 16;
fprintf('\n== STRESS n=4 (concatenacion); %d ciclos ==\n limite      valor    | plano: err mediana  peor | compensado: err mediana  peor | signo ok | excepciones\n', nt);
casos = {};
for e = [-2 -4 -6 -8 -10 -12], casos(end+1, :) = {'singular', -1 - 10^e, 10^e}; end %#ok<AGROW>
for e = [2 4 6 8 10],          casos(end+1, :) = {'rasante',  -1 - 10^e, 10^e}; end %#ok<AGROW>
for e = [-2 -6 -10 -12],       casos(end+1, :) = {'parabol+',  10^e,     10^e}; end %#ok<AGROW>
for k = 1:size(casos, 1)
    e0 = []; e1 = []; ok = 0; nex = 0; primera = '';
    for t = 1:nt
        c = ciclo(n, [0.6, casos{k, 2}, -0.5]); M = rutaMatriz(c);
        try
            W = concatenar(sp, c, rotores(sp, c), saltaciones(sp, c));
            R = W.getIsometryMatrix(); e0(end+1) = norm(R(1:2:end, 1:2:end) - M, 'fro')/norm(M, 'fro'); %#ok<AGROW>
            W.UseCompensatedMapping = true;
            R = W.getIsometryMatrix(); e1(end+1) = norm(R(1:2:end, 1:2:end) - M, 'fro')/norm(M, 'fro'); %#ok<AGROW>
            ok = ok + (signoEspinorial(sp, W) == sign(det(M)));
        catch ex
            nex = nex + 1; if isempty(primera), primera = ex.identifier; end
        end
    end
    if isempty(e0), e0 = NaN; e1 = NaN; end
    fprintf(' %-9s %9.1e | %12.1e %9.1e | %15.1e %9.1e | %4d/%-3d | %d %s\n', casos{k, 1}, casos{k, 3}, median(e0), max(e0), median(e1), max(e1), ok, nt, nex, primera);
end
fprintf('\n== Reproductor minimo del fallo euclideo ==\n');
for n = 2:6
    C = zeros(n); C(1, 2) = 0.3; C(2, 1) = -0.3;
    try
        eu = EuclideanQuadraticSpace.euclidean(n); WE = eu.similitudeFromBivectorCoefficientsMatrix(C, 1.0);
        fprintf(' n=%d, rotacion plana de 0.3 rad en el plano (1,2): ok, %d reflexiones\n', n, WE.Count);
    catch ex
        fprintf(' n=%d, rotacion plana de 0.3 rad en el plano (1,2): FALLA  %s  (%s, linea %d)\n', n, ex.message(1:min(end, 70)), ex.stack(1).name, ex.stack(1).line);
    end
    G = magic(n) - magic(n).'; G = G/norm(G);
    try
        eu = EuclideanQuadraticSpace.euclidean(n); WE = eu.similitudeFromBivectorCoefficientsMatrix(G, 1.0);
        fprintf(' n=%d, antisimetrica generica: ok, %d reflexiones\n', n, WE.Count);
    catch ex
        fprintf(' n=%d, antisimetrica generica: FALLA  (%s, linea %d)\n', n, ex.stack(1).name, ex.stack(1).line);
    end
end
end
% ======================================================================
function c = ciclo(n, sigma)
K = numel(sigma); c.n = n; c.K = K; c.sigma = sigma; c.A = cell(1, K+1); c.Phi = cell(1, K+1); c.U = zeros(n, K); c.Nv = zeros(n, K);
for i = 1:K+1, c.A{i} = (2*rand(n) - 1)*0.6/sqrt(n)*(0.1 + 0.9*rand); c.Phi{i} = expm(c.A{i}); end
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
    p(1:2:end) = c.U(:, k);   p(2:2:end) = c.Nv(:, k);
    q(1:2:end) = c.U(:, k)/s; q(2:2:end) = (1 + 1/s)*c.Nv(:, k);
    w{k} = sp.similitude([p, q], 1.0);
end
end
function W = concatenar(sp, c, wPhi, wS)
% versor total = Phi_K S_K ... S_1 Phi_0 : las normales del factor mas a la izquierda van primero
N = wPhi{1}.getReflectionVectorsMatrix();
for k = 1:c.K, N = [wPhi{k+1}.getReflectionVectorsMatrix(), wS{k}.getReflectionVectorsMatrix(), N]; end %#ok<AGROW>
W = sp.similitude(N, 1.0);
end
function s = signoEspinorial(sp, W)
s = 1; for i = 1:W.Count, s = s*sign(sp.normSquared(W.getReflectionVector(i))); end
end
