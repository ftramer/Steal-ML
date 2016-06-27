SHELL := /bin/bash

BINARY_DATA=moons circles adult_b mushrooms cancer diabetes
MULTI_DATA=blobs class5 iris adult steak gss digits

EXP_DIR=experiments
REGRESSION_DIR=$(EXP_DIR)/LR
BIN_DIR=$(REGRESSION_DIR)/bin
SOFT_DIR=$(REGRESSION_DIR)/softmax
OVR_DIR=$(REGRESSION_DIR)/ovr

ROUNDINGS=5 4 3 2

REPS=4

all: binary binary_results softmax softmax_results ovr ovr_results
clean: binary_clean softmax_clean ovr_clean

binary:
	mkdir -p $(BIN_DIR) ; \
	for data in $(BINARY_DATA) ; do \
		number=1 ; while [[ $$number -le $(REPS) ]] ; do \
			$(MAKE) DATA=$$data SEED=$$number $(BIN_DIR)/$$data.$$number.out ; \
			((number = number + 1)) ; \
		done \
	done

%.out :
ifdef ROUNDING
	python -m regression $(DATA) $(OPTIONS) --rounding $(ROUNDING) --seed $(SEED) >> $@
else
	python -m regression $(DATA) $(OPTIONS) --seed $(SEED) >> $@
endif

softmax:
	mkdir -p $(SOFT_DIR) ; \
	for data in $(MULTI_DATA) ; do \
		number=1 ; while [[ $$number -le $(REPS) ]] ; do \
			$(MAKE) DATA=$$data SEED=$$number OPTIONS='--multinomial' $(SOFT_DIR)/$$data.$$number.out ; \
			((number = number + 1)) ; \
		done \
	done

softmax_rounding:
	for round in $(ROUNDINGS) ; do \
		mkdir -p $(SOFT_DIR)_r$$round ; \
		for data in $(MULTI_DATA) ; do \
			number=1 ; while [[ $$number -le $(REPS) ]] ; do \
				$(MAKE) DATA=$$data SEED=$$number OPTIONS='--multinomial' ROUNDING=$$round $(SOFT_DIR)_r$$round/$$data.$$number.out ; \
				((number = number + 1)) ; \
			done \
		done \
	done

ovr:
	mkdir -p $(OVR_DIR) ; \
	for data in $(MULTI_DATA) ; do \
		number=1 ; while [[ $$number -le $(REPS) ]] ; do \
			$(MAKE) DATA=$$data SEED=$$number $(OVR_DIR)/$$data.$$number.out ; \
			((number = number + 1)) ; \
		done \
	done


results:
	python -m aggregate_results $(DIR) $(DATA)

binary_results:
	$(MAKE) DATA="$(BIN_DIR)/*.out" DIR=$(BIN_DIR) results

softmax_results:
	$(MAKE) DATA="$(SOFT_DIR)/*.out" DIR=$(SOFT_DIR) results

ovr_results:
	$(MAKE) DATA="$(OVR_DIR)/*.out" DIR=$(OVR_DIR) results

binary_clean:
	rm -f $(BIN_DIR)/*.out

softmax_clean:
	rm -f $(SOFT_DIR)/*.out

softmax_rounding_clean:
	for round in $(ROUNDINGS) ; do \
		rm -f $(SOFT_DIR)_r$$round/*.out ; \
	done

ovr_clean:
	rm -f $(OVR_DIR)/*.out
