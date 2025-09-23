"""
Logging configuration and error handling utilities for CLM automation system.
Provides centralized logging setup and error tracking.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import traceback

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    os.makedirs('./logs', exist_ok=True)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
    
    # Error file handler (separate file for errors)
    error_handler = logging.handlers.RotatingFileHandler(
        './logs/errors.log',
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Create specific loggers for different modules
    module_loggers = [
        'clm.database',
        'clm.embeddings', 
        'clm.rag',
        'clm.agent',
        'clm.processor',
        'clm.chatbot'
    ]
    
    for module_name in module_loggers:
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(logging.DEBUG)
    
    # Log system information
    logger = logging.getLogger('clm.system')
    logger.info("=" * 60)
    logger.info("CLM Automation System - Logging Initialized")
    logger.info(f"Log Level: {log_level}")
    logger.info(f"Log File: {log_file or 'Console only'}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Working Directory: {os.getcwd()}")
    logger.info("=" * 60)
    
    return logger

class ErrorHandler:
    """Centralized error handling and reporting"""
    
    def __init__(self):
        self.logger = logging.getLogger('clm.error_handler')
        self.error_counts = {}
    
    def handle_exception(self, 
                        exception: Exception, 
                        context: str = "Unknown", 
                        additional_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Handle exceptions with detailed logging and tracking
        
        Args:
            exception: The exception that occurred
            context: Context where the exception occurred
            additional_data: Additional data for debugging
            
        Returns:
            Error ID for tracking
        """
        error_id = f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(exception)}"
        error_type = type(exception).__name__
        
        # Track error frequency
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        # Log detailed error information
        self.logger.error(f"Error ID: {error_id}")
        self.logger.error(f"Context: {context}")
        self.logger.error(f"Error Type: {error_type}")
        self.logger.error(f"Error Message: {str(exception)}")
        self.logger.error(f"Error Count: {self.error_counts[error_type]}")
        
        if additional_data:
            self.logger.error(f"Additional Data: {additional_data}")
        
        # Log full traceback
        self.logger.error("Full Traceback:")
        self.logger.error(traceback.format_exc())
        
        # Log stack trace for debugging
        stack_trace = traceback.format_stack()
        self.logger.debug("Stack Trace:")
        for line in stack_trace[:-1]:  # Exclude current frame
            self.logger.debug(line.strip())
        
        return error_id
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_errors": total_errors,
            "error_types": dict(self.error_counts),
            "most_common_error": max(self.error_counts, key=self.error_counts.get) if self.error_counts else None
        }
    
    def log_performance_metric(self, 
                             operation: str, 
                             duration: float, 
                             success: bool = True, 
                             additional_metrics: Optional[Dict] = None):
        """Log performance metrics"""
        logger = logging.getLogger('clm.performance')
        
        status = "SUCCESS" if success else "FAILURE"
        logger.info(f"PERF: {operation} - {status} - {duration:.3f}s")
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                logger.info(f"PERF: {operation}.{key} = {value}")

# Global error handler instance
error_handler = ErrorHandler()

def log_function_call(func):
    """Decorator to log function calls with timing and error handling"""
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__qualname__}"
        logger = logging.getLogger('clm.function_calls')
        
        start_time = datetime.now()
        logger.debug(f"CALL: {func_name} started")
        
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"CALL: {func_name} completed successfully in {duration:.3f}s")
            
            # Log performance metric
            error_handler.log_performance_metric(func_name, duration, True)
            
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            # Handle the exception
            error_id = error_handler.handle_exception(
                e, 
                context=f"Function call: {func_name}",
                additional_data={
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                    "duration": duration
                }
            )
            
            logger.error(f"CALL: {func_name} failed after {duration:.3f}s - Error ID: {error_id}")
            
            # Log performance metric for failed call
            error_handler.log_performance_metric(func_name, duration, False)
            
            # Re-raise the exception
            raise
    
    return wrapper

def setup_system_logging():
    """Set up system-wide logging configuration"""
    return setup_logging(
        log_level="INFO",
        log_file="./logs/clm_system.log"
    )

# Health check logging
def log_system_health():
    """Log system health information"""
    logger = logging.getLogger('clm.health')
    
    try:
        import psutil
        
        # Memory usage
        memory = psutil.virtual_memory()
        logger.info(f"HEALTH: Memory usage - {memory.percent}% ({memory.used / 1024**3:.2f}GB used)")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        logger.info(f"HEALTH: Disk usage - {disk.percent}% ({disk.used / 1024**3:.2f}GB used)")
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        logger.info(f"HEALTH: CPU usage - {cpu_percent}%")
        
    except ImportError:
        logger.debug("psutil not available - basic health logging only")
    
    # Log file sizes
    try:
        for log_file in ['./logs/clm_system.log', './logs/errors.log']:
            if os.path.exists(log_file):
                size_mb = os.path.getsize(log_file) / (1024 * 1024)
                logger.info(f"HEALTH: Log file {log_file} - {size_mb:.2f}MB")
    except Exception as e:
        logger.warning(f"Could not check log file sizes: {e}")
    
    # Error statistics
    error_stats = error_handler.get_error_statistics()
    logger.info(f"HEALTH: Total errors since startup - {error_stats['total_errors']}")
    if error_stats['most_common_error']:
        logger.info(f"HEALTH: Most common error - {error_stats['most_common_error']}")

if __name__ == "__main__":
    # Test the logging configuration
    setup_system_logging()
    
    logger = logging.getLogger('clm.test')
    logger.debug("Debug message test")
    logger.info("Info message test")
    logger.warning("Warning message test")
    logger.error("Error message test")
    
    # Test error handling
    try:
        raise ValueError("Test error for logging")
    except Exception as e:
        error_id = error_handler.handle_exception(e, "Testing error handling")
        print(f"Test error handled with ID: {error_id}")
    
    # Test health logging
    log_system_health()
    
    print("Logging test completed - check ./logs/ directory for output")