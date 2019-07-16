using Flux
using Flux: onehot, chunk, batchseq, throttle, crossentropy
using StatsBase: wsample
using Base.Iterators: partition


include("char-rnn.jl")
include("char-lstm.jl")
include("char-gru.jl")

cd(@__DIR__)

isfile("input.txt") ||
  download("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
           "input.txt")

text = collect(String(read("input.txt")))
alphabet = [unique(text)..., '_']
text = map(ch -> onehot(ch, alphabet), text)
stop = onehot('_', alphabet)

N = length(alphabet)
seqlen = 100
nbatch = 30

Xs = collect(partition(batchseq(chunk(text, nbatch), stop), seqlen))
Ys = collect(partition(batchseq(chunk(text[2:end], nbatch), stop), seqlen))

Xs = [Xs;Xs;Xs;Xs]
Ys = [Ys;Ys;Ys;Ys]

opt = ADAM(0.01)
tx, ty = (Xs[5], Ys[5])

function sample(r, alphabet, len; temp = 1)
  r = cpu(r)
  Flux.reset!(r)
  buf = IOBuffer()
  c = rand(alphabet)
  for i = 1:len
    write(buf, c)
    c = wsample(alphabet, r(onehot(c, alphabet)).data)
  end
  return String(take!(buf))
end

timeAff = 10;

println(" ################## RNN TEST ################## ")
r, evacbRnn = charRnn(N, tx, ty, opt)
@time fTrainRnn(timeAff, r, Xs, Ys, opt, evalcbRnn)

sample(r, alphabet, 1000) |> println

println(" ################## LSTM TEST ################## ")
l, evalcbLstm = charLstm(N, tx, ty, opt)
@time fTrainLstm(timeAff, l, Xs, Ys, opt, evalcbLstm)

sample(l, alphabet, 1000) |> println

println(" ################## GRU TEST ################## ")
g, evalcbGru = charGru(N, tx, ty, opt)
@time fTrainGru(timeAff, g, Xs, Ys, opt, evalcbGru)

sample(g, alphabet, 1000) |> println

# evalcb = function ()
#   @show loss(Xs[5], Ys[5])
#   println(sample(deepcopy(m), alphabet, 100))
# end
