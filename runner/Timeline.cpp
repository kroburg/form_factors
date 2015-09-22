#include "Timeline.h"

Timeline::frame_t::frame_t(float step, const vector<float>& temps) : step(step), temps(temps) { }

Timeline::frame_t::frame_t(const frame_t& other) : step(other.step), temps(other.temps) { }

void Timeline::reset() {
    curTime = 0;
    prevStep = 0;
    frames.clear();
}

void Timeline::addFrame(frame_t& frameToAdd) {
    frame_t frame(frameToAdd);
    auto timeToAdd = curTime + prevStep;
    curTime = timeToAdd;
    prevStep = frame.step;
    frames.insert(pair<const float, frame_t>(timeToAdd, frame));
}

Timeline::Timeline(): curTime(0), prevStep(0) { }

float Timeline::getCurTime() const {
    return curTime;
}

vector<float> Timeline::getPoints() const {
    vector<float> result;
    for_each(frames.begin(), frames.end(), [this, &result](pair<const float, frame_t> p) {
        result.push_back(p.first / getCurTime());
    });
    return result;
}

int Timeline::findTempForPos(float pos, vector<float>& temps) {
    auto curPos = pos * getCurTime();
    auto to = frames.upper_bound(curPos);
    if (to != frames.end() && to != frames.begin()) {
        auto from = to;
        --from;
        auto frameFrom = from->second;
        auto frameTo = to->second;
        auto dydx = (curPos - from->first) / (to->first - from->first);
        for (int i = 0; i < std::min(frameFrom.temps.size(), frameTo.temps.size()); ++i) {
            float temp = frameFrom.temps[i] + dydx * (frameTo.temps[i] - frameFrom.temps[i]);
            temps.push_back(temp);
            //printf("%f, %f\n", frameFrom.temps[i], frameTo.temps[i]);
        }
        return 0;
    }
    return -1;
}