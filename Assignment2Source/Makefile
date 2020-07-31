default: all

all:
	exp1 exp2-1 exp2-2 exp3

exp1:
	python3 optimized_neural_exp1.py

exp2-1:
	python3 optimized_neural_exp2_half.py

exp2-2:
	python3 optimized_neural_exp2_quarter.py

exp3:
	python3 optimized_neural_exp3.py

clean-exp1:
	rm train_output_exp1_*
	rm validation_output_exp1_*
	rm graph_exp1_*

clean-exp2-1:
	rm train_output_exp2_*
	rm validation_output_exp2_*
	rm graph_exp2_*

clean-exp2-2:
	rm train_output_exp2_*
	rm validation_output_exp2_*
	rm graph_exp2_*

clean-exp3:
	rm train_output_exp3_*
	rm validation_output_exp3_*
	rm graph_exp3_*

clean-all:
	rm train_*
	rm validation_*
	rm graph*
