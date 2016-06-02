#ifndef OPENCV_CUDA_SAMPLES_TICKMETER_
#define OPENCV_CUDA_SAMPLES_TICKMETER_

class CV_EXPORTS TickMeter
{
public:
    TickMeter();
    void start();
    void stop();

    int64 getTimeTicks() const;
    double getTimeMicro() const;
    double getTimeMilli() const;
    double getTimeSec()   const;
    int64 getCounter() const;

    void reset();
private:
    int64 counter;
    int64 sumTime;
    int64 startTime;
};

std::ostream& operator << (std::ostream& out, const TickMeter& tm);


TickMeter::TickMeter() { reset(); }
int64 TickMeter::getTimeTicks() const { return sumTime; }
double TickMeter::getTimeMicro() const { return  getTimeMilli()*1e3;}
double TickMeter::getTimeMilli() const { return getTimeSec()*1e3; }
double TickMeter::getTimeSec() const { return (double)getTimeTicks()/cv::getTickFrequency();}
int64 TickMeter::getCounter() const { return counter; }
void TickMeter::reset() {startTime = 0; sumTime = 0; counter = 0; }

void TickMeter::start(){ startTime = cv::getTickCount(); }
void TickMeter::stop()
{
    int64 time = cv::getTickCount();
    if ( startTime == 0 )
        return;
    ++counter;
    sumTime += ( time - startTime );
    startTime = 0;
}

std::ostream& operator << (std::ostream& out, const TickMeter& tm) { return out << tm.getTimeSec() << "sec"; }

#endif
