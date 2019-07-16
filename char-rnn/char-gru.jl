function charGru(N, tx, ty, opt)

  g = Chain(
    GRU(N, 128),
    GRU(128, 128),
    Dense(128, N),
    softmax)

  g = gpu(g)

  evalcbGru = () -> @show lossGru(tx, ty)
  return g, evalcbGru
end

function lossGru(xs, ys)
  l = sum(crossentropy.(g.(gpu.(xs)), gpu.(ys)))
  Flux.truncate!(g)
  return l
end

fTrainGru(ep, g, Xs, Ys, opt, evalcbGru) = Flux.train!(lossGru, params(g), zip(Xs, Ys), opt,
            cb = throttle(evalcbGru, ep))
