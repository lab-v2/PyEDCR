from PyEDCR import EDCR

edcr = EDCR(epsilon=0.1,
            check_mode=True
            )
edcr.print_metrics(test=False, prior=True)
edcr.print_metrics(test=True, prior=True)

edcr.test_get_predictions()
print("test_get_predictions passed!")