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

cd(@__DIR__)

isfile("input.txt") ||
  download("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
           "input.txt")




function cleanText(text ; upCase = false ,
            number = false, accent = true, word = true )
  if !upCase
    text = replace(text, text => lowercase)
  end
  text = replace(text, r"[;:+†…«»—]" => ' ')
  text = replace(text, r"[œ]" => "oe")
  text = replace(text, r"\n|\\|\t" => ' ')
  text = replace(text, r"[\[{]" => '(')
  text = replace(text, r"[\]}]" => ')')
  if !number
    text = replace(text, r"[1234567890]" => "")
  end
  if !accent
    text = replace(text, r"[âäà]" => 'a')
    text = replace(text, r"[êëéè]" => 'e')
    text = replace(text, r"[îïì]" => 'i')
    text = replace(text, r"[ôöò]" => 'o')
    text = replace(text, r"[ûüù]" => 'u')
    text = replace(text, r"[ç]" => 'c')
  end
  text = replace(text, r"  " => ' ')
  text = replace(text, r"   " => ' ')
  text = replace(text, r"    " => ' ')
  if word
    text = replace(text, ' ' => "; ;")
    text = split(text,r";")
    alphabet = " ";
    pas = 1000;
    to = pas;
    alphabet = unique([alphabet ; text[1:to]])
    while to+pas < length(text)
      alphabet = unique([alphabet ; text[to+1:to+pas]])
      to += pas
    end
    alphabet = unique([alphabet ; text[to+1:length(text)]])
    alphabet = [alphabet ; "_"]
  else
    text = collect(text)
    alphabet = [unique(text)..., '_']
  end
  return text, alphabet
end
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



print("end")

destroy!(ctx)        # nettoyage
# evalcb = function ()
#   @show loss(Xs[5], Ys[5])
#   println(sample(deepcopy(m), alphabet, 100))
# end
