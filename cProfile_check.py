import pstats

p = pstats.Stats('output.pstats')
p.sort_stats('time').print_stats(10)  # Показывает топ-10 медленных функций
