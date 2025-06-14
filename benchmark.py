import torch
import time
from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import psutil
import gc

class BenchmarkDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        # Mix of different text lengths for more realistic testing
        self.texts = [
            "Great product!",
            "This is absolutely terrible and I hate it.",
            "The movie was okay, not great but not bad either. Would probably watch again.",
            "Fantastic experience! Highly recommend to everyone. Five stars!",
            "Poor quality, broke after one day. Very disappointed with this purchase."
        ]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
        return self.texts[i % len(self.texts)]

def get_gpu_info():
    """Get GPU information"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return gpu_name, gpu_memory
    return None, 0

def get_cpu_info():
    """Get CPU information"""
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores
    cpu_freq = psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown"
    return cpu_count, cpu_freq

def benchmark_device(device, dataset, batch_sizes=[1, 8, 32], task="text-classification"):
    """Benchmark a specific device"""
    print(f"\n{'='*50}")
    print(f"Testing on: {'GPU' if device == 0 else 'CPU'}")
    print(f"{'='*50}")
    
    try:
        # Create pipeline
        pipe = pipeline(task, device=device)
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")
            
            # Clear memory
            if device == 0:
                torch.cuda.empty_cache()
            gc.collect()
            
            # Get initial memory
            if device == 0:
                initial_memory = torch.cuda.memory_allocated(0) / 1e6  # MB
            
            # Time the inference
            start_time = time.time()
            
            processed = 0
            with tqdm(total=len(dataset), desc=f"Processing") as pbar:
                for out in pipe(dataset, batch_size=batch_size):
                    processed += 1
                    pbar.update(1)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate metrics
            texts_per_second = len(dataset) / total_time
            time_per_text = (total_time / len(dataset)) * 1000  # milliseconds
            
            # Memory usage
            if device == 0:
                peak_memory = torch.cuda.max_memory_allocated(0) / 1e6  # MB
                memory_used = peak_memory - initial_memory
            else:
                memory_used = "N/A"
            
            results[batch_size] = {
                'total_time': total_time,
                'texts_per_second': texts_per_second,
                'time_per_text': time_per_text,
                'memory_used': memory_used
            }
            
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Texts/second: {texts_per_second:.1f}")
            print(f"  Time per text: {time_per_text:.2f}ms")
            if device == 0:
                print(f"  GPU memory used: {memory_used:.1f}MB")
        
        return results
        
    except Exception as e:
        print(f"Error testing on {'GPU' if device == 0 else 'CPU'}: {e}")
        return None

def main():
    print("🚀 GPU vs CPU Benchmark for Text Classification")
    print("=" * 60)
    
    # System info
    gpu_name, gpu_memory = get_gpu_info()
    cpu_cores, cpu_freq = get_cpu_info()
    
    print(f"🖥️  CPU: {cpu_cores} cores @ {cpu_freq/1000:.1f}GHz" if cpu_freq != "Unknown" else f"🖥️  CPU: {cpu_cores} cores")
    if gpu_name:
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        gpu_available = True
    else:
        print("🎮 GPU: Not available")
        gpu_available = False
    
    # Create dataset
    dataset = BenchmarkDataset(size=500)  # Smaller for quicker testing
    batch_sizes = [1, 8, 32]
    
    # Test CPU
    cpu_results = benchmark_device(-1, dataset, batch_sizes)
    
    # Test GPU if available
    gpu_results = None
    if gpu_available:
        gpu_results = benchmark_device(0, dataset, batch_sizes)
    
    # Compare results
    if cpu_results and gpu_results:
        print(f"\n{'='*60}")
        print("📊 PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        
        print(f"{'Batch Size':<12} {'CPU (txt/s)':<12} {'GPU (txt/s)':<12} {'Speedup':<10}")
        print("-" * 50)
        
        for batch_size in batch_sizes:
            if batch_size in cpu_results and batch_size in gpu_results:
                cpu_speed = cpu_results[batch_size]['texts_per_second']
                gpu_speed = gpu_results[batch_size]['texts_per_second']
                speedup = gpu_speed / cpu_speed
                
                print(f"{batch_size:<12} {cpu_speed:<12.1f} {gpu_speed:<12.1f} {speedup:<10.1f}x")
        
        # Best performance
        best_cpu = max(cpu_results.values(), key=lambda x: x['texts_per_second'])
        best_gpu = max(gpu_results.values(), key=lambda x: x['texts_per_second'])
        
        print(f"\n🏆 Best CPU Performance: {best_cpu['texts_per_second']:.1f} texts/second")
        print(f"🏆 Best GPU Performance: {best_gpu['texts_per_second']:.1f} texts/second")
        print(f"🚀 Overall GPU Speedup: {best_gpu['texts_per_second']/best_cpu['texts_per_second']:.1f}x faster")
    
    elif cpu_results:
        print(f"\n📊 CPU-only results available")
        best_cpu = max(cpu_results.values(), key=lambda x: x['texts_per_second'])
        print(f"🏆 Best CPU Performance: {best_cpu['texts_per_second']:.1f} texts/second")

if __name__ == "__main__":
    main()