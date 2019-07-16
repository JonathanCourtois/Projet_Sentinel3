function charLstm(N, tx, ty, opt)
  l = Chain(
    LSTM(N, 128),
    LSTM(128, 128),
    Dense(128, N),
    softmax)

  l = gpu(l)

  evalcbLstm = () -> @show lossLstm(tx, ty)
  return l, evalcbLstm
end

function lossLstm(xs, ys)
  lo = sum(crossentropy.(l.(gpu.(xs)), gpu.(ys)))
  Flux.truncate!(l)
  return lo
end

fTrainLstm(ep, l, Xs, Ys, opt, evalcbLstm) = Flux.train!(lossLstm, params(l), zip(Xs, Ys), opt,
            cb = throttle(evalcbLstm, ep))
