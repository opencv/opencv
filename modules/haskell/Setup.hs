import Distribution.Simple

main :: IO ()
main = defaultMainWithHooks simpleUserHooks {
                                            -- The following hook prevents cabal from checking to see if generated cpp header and
                                            -- source files actually compile before completing configuration.
                                            postConf = postConf emptyUserHooks } 

