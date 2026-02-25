#include "llvm/Support/Debug.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

namespace mlir {

static std::string format(double d) {
  char x[20];
  sprintf(x, "%.3f", d);
  return x;
}

static std::string formatTime(double seconds) {
  if (seconds >= 1.0)
    return format(seconds) + " s";
  if (seconds >= 1e-3)
    return format(seconds * 1e3) + " ms";

  return format(seconds * 1e6) + " us";
}

class Profiler {
public:
  enum StartKind {
    NONE, DEPS, COMPMISS, DIST, COUNT,
  };
private:
  // In seconds.
  struct SinkProfile {
    double depsTime;
    double compMissTime;

    std::vector<double> distTimes;
    std::vector<double> countTimes;

    int lexIndex;
  };

  std::vector<SinkProfile> sinks;
  llvm::Timer timer;
  StartKind kind;
public:
  SinkProfile &current() {
    return sinks.back();
  }

  void addSink(int lexIndex) {
    sinks.push_back(SinkProfile{});
    current().lexIndex = lexIndex;
  }

  void start(StartKind k) {
    if (kind != NONE && timer.isRunning())
      end();
    kind = k;
    timer.startTimer();
  }

  void end() {
    timer.stopTimer();
    double time = timer.getTotalTime().getWallTime();
    timer.clear();
    switch (kind) {
    case DEPS:
      current().depsTime = time;
      break;
    case COMPMISS:
      current().compMissTime = time;
      break;
    case DIST:
      current().distTimes.push_back(time);
      break;
    case COUNT:
      current().countTimes.push_back(time);
      break;
    default:
      llvm_unreachable("Misconfigured profiler");
    }
  }

  void report() {
    SinkProfile &record = current();
    llvm::dbgs() << "===== LEX INDEX " << record.lexIndex << " START =====\n";
    llvm::dbgs() << "Deps time: " << formatTime(record.depsTime) << "\n";
    llvm::dbgs() << "Comp. miss time: " << formatTime(record.compMissTime) << "\n\n";

    if (record.distTimes.size() > 0) {
      llvm::dbgs() << "Pieces: " << record.distTimes.size() << "\n";
      auto totalDist = std::accumulate(record.distTimes.begin(), record.distTimes.end(), 0.0);
      auto maxDist = std::max_element(record.distTimes.begin(), record.distTimes.end());
      llvm::dbgs() << "Total dist time: " << formatTime(totalDist) << "\n";
      llvm::dbgs() << "Average dist time: " << formatTime(totalDist / record.distTimes.size())<< "\n";
      llvm::dbgs() << "Maximum dist time: " << formatTime(*maxDist) << "\n\n";
    }

    if (record.countTimes.size() > 0) {
      auto totalCount = std::accumulate(record.countTimes.begin(), record.countTimes.end(), 0.0);
      auto maxCount = std::max_element(record.countTimes.begin(), record.countTimes.end());
      llvm::dbgs() << "Total count time: " << formatTime(totalCount) << "\n";
      llvm::dbgs() << "Average count time: " << formatTime(totalCount / record.countTimes.size()) << "\n";
      llvm::dbgs() << "Maximum count time: " << formatTime(*maxCount) << "\n";
      llvm::dbgs() << "===== LEX INDEX " << record.lexIndex << " END =====\n";
    }
  }
} extern profiler;

} // namespace mlir
