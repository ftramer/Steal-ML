SHELL := /bin/bash

DATA=tax steak gss email spam med_price bitcoin credit

EXP_DIR=experiments

REPS=1

ROUNDINGS=5 4 3 2

HEADER='dataset,method,leaves,unique,loss,loss_u,budget,time(s)'

%.out:
ifdef ROUNDING
	python -m bigml_wrapper.tree $(DATA) --rounding $(ROUNDING) $(OPTIONS) >> $@
else
	python -m bigml_wrapper.tree $(DATA) $(OPTIONS) >> $@
endif

bottom:
	mkdir -p $(EXP_DIR)
	for data in $(DATA) ; do \
		$(MAKE) DATA=$$data $(EXP_DIR)/$$data.bottom.out ; \
	done

top:
	mkdir -p $(EXP_DIR)
	for data in $(DATA) ; do \
		$(MAKE) DATA=$$data OPTIONS='--top' $(EXP_DIR)/$$data.top.out ; \
	done

top_rounding:
	mkdir -p $(EXP_DIR)
	for round in $(ROUNDINGS) ; do \
		for data in $(DATA) ; do \
			$(MAKE) DATA=$$data  OPTIONS='--top' ROUNDING=$$round $(EXP_DIR)/$$data.top.r$$round.out ; \
		done \
	done


clean_bottom:
	rm -f $(EXP_DIR)/*_bottom.out

clean_top:
	rm -f $(EXP_DIR)/*_top.out