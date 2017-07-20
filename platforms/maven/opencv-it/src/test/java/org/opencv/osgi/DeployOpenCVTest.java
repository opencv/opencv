package org.opencv.osgi;

import java.io.File;
import javax.inject.Inject;
import junit.framework.TestCase;
import org.apache.karaf.log.core.LogService;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.ops4j.pax.exam.Configuration;
import static org.ops4j.pax.exam.CoreOptions.maven;
import static org.ops4j.pax.exam.CoreOptions.mavenBundle;
import org.ops4j.pax.exam.Option;
import org.ops4j.pax.exam.junit.PaxExam;
import static org.ops4j.pax.exam.karaf.options.KarafDistributionOption.karafDistributionConfiguration;
import static org.ops4j.pax.exam.karaf.options.KarafDistributionOption.keepRuntimeFolder;
import static org.ops4j.pax.exam.karaf.options.KarafDistributionOption.logLevel;
import org.ops4j.pax.exam.karaf.options.LogLevelOption;
import org.ops4j.pax.exam.options.MavenArtifactUrlReference;
import org.ops4j.pax.exam.spi.reactors.ExamReactorStrategy;
import org.ops4j.pax.exam.spi.reactors.PerClass;
import org.ops4j.pax.logging.spi.PaxLoggingEvent;
import org.osgi.framework.BundleContext;

/**
 *
 * @author Kerry Billingham <contact@AvionicEngineers.com>
 */
@ExamReactorStrategy(PerClass.class)
@RunWith(PaxExam.class)
public class DeployOpenCVTest {

    /*
    The expected string in Karaf logs when the bundle has deployed and native library loaded.
    */
    private static final String OPENCV_SUCCESSFUL_LOAD_STRING = "Successfully loaded OpenCV native library.";

    private static final String KARAF_VERSION = "4.0.6";

    @Inject
    protected BundleContext bundleContext;

    @Inject
    private LogService logService;

    /*
    This service is required to ensure that the native library has been loaded
    before any test is carried out.
    */
    @Inject
    private OpenCVInterface openCVInterface;

    @Configuration
    public static Option[] configuration() throws Exception {
        MavenArtifactUrlReference karafUrl = maven()
                .groupId("org.apache.karaf")
                .artifactId("apache-karaf")
                .version(KARAF_VERSION)
                .type("tar.gz");
        return new Option[]{
            karafDistributionConfiguration()
            .frameworkUrl(karafUrl)
            .unpackDirectory(new File("../../../build/target/exam"))
            .useDeployFolder(false),
            keepRuntimeFolder(),
            mavenBundle()
            .groupId("org.opencv")
            .artifactId("opencv")
            .version("3.3.0"),
            logLevel(LogLevelOption.LogLevel.INFO)
        };
    }

    /**
     * Tests that the OpenCV bundle has been successfully deployed and that the
     * native library has been loaded.
     */
    @Test
    public void testOpenCVNativeLibraryLoadSuccess() {

        Iterable<PaxLoggingEvent> loggingEvents = logService.getEvents();
        boolean loadSuccessful = logsContainsMessage(loggingEvents, OPENCV_SUCCESSFUL_LOAD_STRING);

        TestCase.assertTrue("Could not determine if OpenCV library successfully loaded from the logs.", loadSuccessful);

    }

    private boolean logsContainsMessage(Iterable<PaxLoggingEvent> logEnumeration, final String logMessageString) {
        boolean contains = false;
        for (PaxLoggingEvent logEntry : logEnumeration) {
            if (logEntry.getMessage().contains(logMessageString)) {
                contains = true;
                break;
            }
        }
        return contains;
    }
}
