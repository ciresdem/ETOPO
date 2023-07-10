<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis hasScaleBasedVisibilityFlag="0" version="3.22.16-Białowieża" styleCategories="AllStyleCategories" maxScale="0" minScale="1e+08">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
    <Private>0</Private>
  </flags>
  <temporal mode="0" enabled="0" fetchMode="0">
    <fixedRange>
      <start></start>
      <end></end>
    </fixedRange>
  </temporal>
  <customproperties>
    <Option type="Map">
      <Option name="WMSBackgroundLayer" value="false" type="bool"/>
      <Option name="WMSPublishDataSourceUrl" value="false" type="bool"/>
      <Option name="embeddedWidgets/count" value="0" type="int"/>
      <Option name="identify/format" value="Value" type="QString"/>
    </Option>
  </customproperties>
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option name="name" value="" type="QString"/>
      <Option name="properties"/>
      <Option name="type" value="collection" type="QString"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling zoomedInResamplingMethod="nearestNeighbour" zoomedOutResamplingMethod="nearestNeighbour" maxOversampling="2" enabled="false"/>
    </provider>
    <rasterrenderer classificationMin="-8000" opacity="1" classificationMax="1000" type="singlebandpseudocolor" nodataColor="" band="1" alphaBand="-1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader clip="0" labelPrecision="4" minimumValue="-8000" colorRampType="INTERPOLATED" classificationMode="1" maximumValue="1000">
          <colorramp name="[source]" type="gradient">
            <Option type="Map">
              <Option name="color1" value="18,50,211,255" type="QString"/>
              <Option name="color2" value="255,255,255,255" type="QString"/>
              <Option name="discrete" value="0" type="QString"/>
              <Option name="rampType" value="gradient" type="QString"/>
              <Option name="stops" value="0.333333;12,184,153,255:0.666667;223,200,49,255:0.777778;255,81,0,255:0.833333;255,0,255,255:0.866667;153,0,255,255:0.877778;0,12,255,255:0.888878;137,206,255,255:0.888889;17,132,0,255:0.9;106,219,83,255:0.911111;255,247,83,255:0.933333;253,174,97,255:0.972222;215,25,28,255" type="QString"/>
            </Option>
            <prop k="color1" v="18,50,211,255"/>
            <prop k="color2" v="255,255,255,255"/>
            <prop k="discrete" v="0"/>
            <prop k="rampType" v="gradient"/>
            <prop k="stops" v="0.333333;12,184,153,255:0.666667;223,200,49,255:0.777778;255,81,0,255:0.833333;255,0,255,255:0.866667;153,0,255,255:0.877778;0,12,255,255:0.888878;137,206,255,255:0.888889;17,132,0,255:0.9;106,219,83,255:0.911111;255,247,83,255:0.933333;253,174,97,255:0.972222;215,25,28,255"/>
          </colorramp>
          <item color="#1232d3" alpha="255" value="-8000" label="-8000.0000"/>
          <item color="#0cb899" alpha="255" value="-5000" label="-5000.0000"/>
          <item color="#dfc831" alpha="255" value="-2000" label="-2000.0000"/>
          <item color="#ff5100" alpha="255" value="-1000" label="-1000.0000"/>
          <item color="#ff00ff" alpha="255" value="-500" label="-500.0000"/>
          <item color="#9900ff" alpha="255" value="-200" label="-200.0000"/>
          <item color="#000cff" alpha="255" value="-100" label="-100.0000"/>
          <item color="#89ceff" alpha="255" value="-0.1" label="-0.1000"/>
          <item color="#118400" alpha="255" value="0" label="0.0000"/>
          <item color="#6adb53" alpha="255" value="100" label="100.0000"/>
          <item color="#fff753" alpha="255" value="200" label="200.0000"/>
          <item color="#fdae61" alpha="255" value="400" label="400.0000"/>
          <item color="#d7191c" alpha="255" value="750" label="750.0000"/>
          <item color="#ffffff" alpha="255" value="1000" label="1000.0000"/>
          <rampLegendSettings useContinuousLegend="1" direction="0" maximumLabel="" prefix="" orientation="2" minimumLabel="" suffix="">
            <numericFormat id="basic">
              <Option type="Map">
                <Option name="decimal_separator" value="" type="QChar"/>
                <Option name="decimals" value="6" type="int"/>
                <Option name="rounding_type" value="0" type="int"/>
                <Option name="show_plus" value="false" type="bool"/>
                <Option name="show_thousand_separator" value="true" type="bool"/>
                <Option name="show_trailing_zeros" value="false" type="bool"/>
                <Option name="thousand_separator" value="" type="QChar"/>
              </Option>
            </numericFormat>
          </rampLegendSettings>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast contrast="0" gamma="1" brightness="0"/>
    <huesaturation colorizeRed="255" colorizeOn="0" colorizeGreen="128" invertColors="0" saturation="0" grayscaleMode="0" colorizeBlue="128" colorizeStrength="100"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
