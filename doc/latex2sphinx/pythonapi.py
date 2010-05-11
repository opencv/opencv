class Argument:
  def __init__(self, fields):
    self.ty = fields[0]
    self.nm = fields[1]
    self.flags = ""
    self.init = None

    if len(fields) > 2:
      if fields[2][0] == '/':
        self.flags = fields[2][1:].split(",")
      else:
        self.init = fields[2]

def reader(apifile):
  api = []
  for l in open(apifile):
    if l[0] == '#':
      continue
    l = l.rstrip()
    f = l.split()
    if len(f) != 0:
      if l[0] != ' ':
        if len(f) > 1:
          ty = f[1]
        else:
          ty = None
        api.append((f[0], [], ty))
      else:
        api[-1][1].append(Argument(f))
  return dict([(a, (ins, outs)) for (a, ins, outs) in api])
