function charRnn(N, tx, ty, opt)

  r = Chain(
    RNN(N, 128),
    RNN(128, 128),
    Dense(128, N),
    softmax)

  r = gpu(r)
  errorRnn = Float64[];
  evalcbRnn = () -> push!(errorRnn, Tracker.data(lossRnn(tx, ty)))
  return r, evalcbRnn, errorRnn
end

function lossRnn(xs, ys)
  l = sum(crossentropy.(r.(gpu.(xs)), gpu.(ys)))
  Flux.truncate!(r)
  return l
end

fTrainRnn(r, Xs, Ys, opt, evalcbRnn) = Flux.train!(lossRnn, params(r), zip(Xs, Ys), opt,
            cb = evalcbRnn)
