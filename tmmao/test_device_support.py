import torch
import logging
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import optics_example

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_test(device_type):
    """Run test with specified device type"""
    device = torch.device(device_type)
    logger.info(f'Testing with {device_type} device...')
    try:
        result = optics_example.main(verbose=1, device=device)
        logger.info(f'{device_type} test completed successfully. Final result: {result}')
        return result
    except Exception as e:
        logger.error(f'Error during {device_type} test: {str(e)}')
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Main test function"""
    # Create output directory
    os.makedirs('output', exist_ok=True)

    # Test with CPU
    result_cpu = run_test('cpu')

    # Test with CUDA if available
    result_cuda = None
    if torch.cuda.is_available():
        result_cuda = run_test('cuda')

        if result_cpu is not None and result_cuda is not None:
            # Compare results (allowing for small numerical differences)
            logger.info('Comparing CPU and CUDA results...')
            if abs(result_cpu - result_cuda) < 1e-5:
                logger.info('Results match within tolerance!')
            else:
                logger.warning(f'Results differ: CPU={result_cpu}, CUDA={result_cuda}')

    return result_cpu is not None and (not torch.cuda.is_available() or result_cuda is not None)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
