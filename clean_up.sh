for i in results/gru_exp*; do zip -r -qq $i.zip $i; rm -rf $i; done
