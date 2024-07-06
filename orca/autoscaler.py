from celery.worker.autoscale import Autoscaler as CeleryAutoscaler

class CalimScaler(CeleryAutoscaler):
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