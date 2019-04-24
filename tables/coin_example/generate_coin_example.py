def p_unfair(h, t, n, m, p_H):
  part1 = m*(p_H**h)*((1.-p_H)**t)
  part2 = n / (2.**(h+t))
  return part1 / (part1+part2)

n = 50
m = 50
print('m = {m:d}, n = {n:d}'.format(m=m, n=n))

with open('coin_example.tex', 'w') as f_table:
  column_names = ['h', 't', 'p_H', 'p_unfair']
  ncol = len(column_names)

  f_table.write('% $m = {m:d}$, $n = {n:d}$\n\n'.format(m=m ,n=n))
  f_table.write('\\begin{tabular}{')
  for icolumn_name in range(ncol):
    f_table.write('c')
    if icolumn_name != ncol-1:
      f_table.write(' ')
  f_table.write('}\n')
  f_table.write('\hline\n')
  f_table.write('$h$ & $t$ & $p_{H}$ & $P\left(\stcomp{F} \mid h,t\\right)$')
  f_table.write(' \\\\\n')
  f_table.write('\hline\n')
  f_table.write('\hline\n')

  for (h,t) in zip([10, 7, 5, 75], [0, 3, 5, 25]):
    for p_H in [1., 0.8, 0.6]:
      if t != 0 and p_H == 1.:
        p_H = 0.99 # might as well change from 1.0 when t != 0 since the P is obviously then 0.0
      result = '{h:d}, {t:d}, {p_H:.2f}, {p_unfair:.5f}'.format(h=h, t=t, p_H=p_H, p_unfair=p_unfair(h, t, n, m, p_H))
      print(result)
      f_table.write('{0:s} \\\\\n'.format(result.replace(',', ' &')))
    f_table.write('\hline\n')
  f_table.write('\end{tabular}')
  f_table.close()
