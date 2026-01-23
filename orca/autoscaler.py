"""Custom Celery autoscaler for the OVRO-LWA Cluster.

Provides a custom autoscaling strategy that considers system load
average and available RAM when scaling worker processes.

Classes
-------
CalimScaler
    Autoscaler that scales workers based on load and memory constraints.
"""
from celery.worker.autoscale import Autoscaler as CeleryAutoscaler


class CalimScaler(CeleryAutoscaler):
    """Custom Celery autoscaler for resource-aware worker scaling.

    Extends the default Celery autoscaler to consider system load average
    and available RAM when making scaling decisions. This prevents
    over-provisioning on shared compute nodes.

    Scaling up occurs when:
        - More tasks are queued than processes available
        - System load average is below 20
        - Available RAM exceeds 200 GB

    Scaling down occurs when:
        - Fewer tasks than processes, or load is acceptable

    Note
    ----
    Cleanup and retry logic may need further implementation.
    """

    def  _maybe_scale(self, req=None) -> bool:
        # Scale up the pool if self.qty > self.processes and load_average < 20 and available RAM > 200 GB.
        if self.qty > self.processes and self.load_average() < 20 and self.available_ram() > 200:
            self.scale_up()
            return True
        
        if self.qty < self.processes or self.load_average() < 20:
            self.scale_down()
            return True
        return False

    # probably need to figure out clean up and retries