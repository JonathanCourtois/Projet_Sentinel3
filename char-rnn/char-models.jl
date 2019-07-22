using Flux
using Flux: onehot, chunk, batchseq, throttle, crossentropy
using StatsBase: wsample
using Base.Iterators: partition
using CuArrays
using CUDAdrv

CUDAdrv.name(CuDevice(1)) 	# voir le GPU 1
dev = CuDevice(1)				# choisir le GPU 1
ctx = CuContext(dev)

include("char-rnn.jl")
include("char-lstm.jl")
include("char-gru.jl")
include("textCleaner.jl")

cd(@__DIR__)

isfile("input.txt") ||
  download("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
           "input.txt")

text = String(read("input.txt"))

text, alphabet = cleanText(text; word = true)

text = map(ch -> onehot(ch, alphabet), text)
stop = onehot("_", alphabet)

N = length(alphabet)
seqlen = 100
nbatch = 30

Xs = collect(partition(batchseq(chunk(text, nbatch), stop), seqlen))
Ys = collect(partition(batchseq(chunk(text[2:end], nbatch), stop), seqlen))

opt = ADAM(0.01)
tx, ty = (Xs[1], Ys[1])

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


println(" ################## RNN TEST ################## ")
r, evacbRnn, errorRnn = charRnn(N, tx, ty, opt)
@time r = fTrainRnn(r, Xs, Ys, opt, evalcbRnn)

sample(r, alphabet, 1000) |> println

println(" ################## LSTM TEST ################## ")
l, evalcbLstm, errorLstm = charLstm(N, tx, ty, opt)
@time l = fTrainLstm(l, Xs, Ys, opt, evalcbLstm)

sample(l, alphabet, 1000) |> println

println(" ################## GRU TEST ################## ")
g, evalcbGru, errorGru = charGru(N, tx, ty, opt)
@time g = fTrainGru(g, Xs, Ys, opt, evalcbGru)

sample(g, alphabet, 1000) |> println



print("end")

destroy!(ctx)        # nettoyage
# evalcb = function ()
#   @show loss(Xs[5], Ys[5])
#   println(sample(deepcopy(m), alphabet, 100))
# end
